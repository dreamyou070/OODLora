from typing import List, Generic, TypeVar, Callable, Union, Any
import functools
import itertools
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
import torch.nn as nn
from pathlib import Path
from .utils import cache_dir, auto_autocast

__all__ = ['ObjectHooker', 'ModuleLocator', 'AggregateHooker', 'UNetCrossAttentionLocator']


ModuleType = TypeVar('ModuleType')
ModuleListType = TypeVar('ModuleListType', bound=List)


class ModuleLocator(Generic[ModuleType]):
    def locate(self, model: nn.Module) -> List[ModuleType]:
        raise NotImplementedError


class ObjectHooker(Generic[ModuleType]):
    def __init__(self, module: ModuleType):
        self.module: ModuleType = module
        self.hooked = False
        self.old_state = dict()

    def __enter__(self):
        self.hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unhook()

    def hook(self):
        if self.hooked:
            raise RuntimeError('Already hooked module')

        self.old_state = dict()
        self.hooked = True
        self._hook_impl()

        return self

    def unhook(self):
        if not self.hooked:
            raise RuntimeError('Module is not hooked')

        for k, v in self.old_state.items():
            if k.startswith('old_fn_'):
                setattr(self.module, k[7:], v)

        self.hooked = False
        self._unhook_impl()

        return self

    def monkey_patch(self, fn_name, fn):
        self.old_state[f'old_fn_{fn_name}'] = getattr(self.module, fn_name)
        setattr(self.module, fn_name, functools.partial(fn, self.module))

    def monkey_super(self, fn_name, *args, **kwargs):
        return self.old_state[f'old_fn_{fn_name}'](*args, **kwargs)

    def _hook_impl(self):
        raise NotImplementedError

    def _unhook_impl(self):
        pass


class AggregateHooker(ObjectHooker[ModuleListType]):
    def _hook_impl(self):
        for h in self.module:
            h.hook()

    def _unhook_impl(self):
        for h in self.module:
            h.unhook()

    def register_hook(self, hook: ObjectHooker):
        self.module.append(hook)


class UNetCrossAttentionLocator(ModuleLocator[Attention]):
    def __init__(self, restrict: bool = None, locate_middle_block: bool = False):
        self.restrict = restrict
        self.layer_names = []
        self.locate_middle_block = locate_middle_block

    def locate(self, model: UNet2DConditionModel) -> List[Attention]:
        """
        Locate all cross-attention modules in a UNet2DConditionModel.

        Args:
            model (`UNet2DConditionModel`): The model to locate the cross-attention modules in.

        Returns:
            `List[Attention]`: The list of cross-attention modules.
        """
        self.layer_names.clear()
        blocks_list = []
        up_names = ['up'] * len(model.up_blocks)
        down_names = ['down'] * len(model.down_blocks)

        for unet_block, name in itertools.chain(zip(model.up_blocks, up_names),zip(model.down_blocks, down_names),
                                                zip([model.mid_block], ['mid']) if self.locate_middle_block else [],):
            if 'CrossAttn' in unet_block.__class__.__name__:
                blocks = []
                for spatial_transformer in unet_block.attentions:
                    for transformer_block in spatial_transformer.transformer_blocks:
                        blocks.append(transformer_block.attn2)
                blocks = [b for idx, b in enumerate(blocks) if self.restrict is None or idx in self.restrict]
                names = [f'{name}-attn-{i}' for i in range(len(blocks)) if self.restrict is None or i in self.restrict]
                blocks_list.extend(blocks)
                self.layer_names.extend(names)

        return blocks_list

class UNetCrossAttentionHooker(ObjectHooker[Attention]):
    def __init__(
            self,
            module: Attention,
            parent_trace: 'trace',
            context_size: int = 77,
            layer_idx: int = 0,
            latent_hw: int = 9216,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: Union[str, Path] = None,
    ):
        super().__init__(module)
        self.heat_maps = parent_trace.all_heat_maps
        self.context_size = context_size
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw

        self.load_heads = load_heads
        self.save_heads = save_heads
        self.trace = parent_trace

        if data_dir is not None:
            data_dir = Path(data_dir)
        else:
            data_dir = cache_dir() / 'heads'

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _unravel_attn(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)
        with auto_autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.view(map_.size(0), h, w)
                map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
                maps.append(map_)
        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        return maps.permute(1, 0, 2, 3).contiguous()  # shape: (heads, tokens, height, width)

    def _save_attn(self, attn_slice: torch.Tensor):
        torch.save(attn_slice, self.data_dir / f'{self.trace._gen_idx}.pt')

    def _load_attn(self) -> torch.Tensor:
        return torch.load(self.data_dir / f'{self.trace._gen_idx}.pt')

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
    ):
        """Capture attentions and aggregate them."""
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # DAAM save heads
        if self.save_heads:
            self._save_attn(attention_probs)
        elif self.load_heads:
            attention_probs = self._load_attn()

        # compute shape factor
        factor = int(math.sqrt(self.latent_hw // attention_probs.shape[1]))
        self.trace._gen_idx += 1

        # skip if too large
        if attention_probs.shape[-1] == self.context_size and factor != 8:
            # shape: (batch_size, 64 // factor, 64 // factor, 77)
            # not mid block
            maps = self._unravel_attn(attention_probs)
            for head_idx, heatmap in enumerate(maps):
                self.heat_maps.update(factor, self.layer_idx, head_idx, heatmap)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def _hook_impl(self):
        self.module.set_processor(self)

unetcrossattnhooker = UNetCrossAttentionHooker(module=,
                                               parent_trace: 'trace',
                                               context_size: int = 77,
                                               layer_idx: int = 0,
                                               latent_hw: int = 9216,
                                               load_heads: bool = False,
                                               save_heads: bool = False,
                                               data_dir: Union[str, Path] = None,)