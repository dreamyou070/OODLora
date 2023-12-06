import torch

def register_attention_control(unet, controller):

    def ca_forward(self, layer_name):
        def forward(hidden_states, context=None, mask=None):
            is_cross_attention = False
            if context is not None:
                is_cross_attention = True
            batch_size, sequence_length, _ = hidden_states.shape
            query = self.to_q(hidden_states)
            context = context if context is not None else hidden_states
            key = self.to_k(context)
            value = self.to_v(context)
            dim = query.shape[-1]
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            # 1) attention score
            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query, key.transpose(-1, -2), beta=0, alpha=self.scale, )
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(value.dtype)
            if is_cross_attention:
                attn = controller.store(attention_probs, layer_name)
            # 2) after value calculating
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            #print(f'self.to_out {self.to_out}')
            hidden_states = self.to_out[0](hidden_states)
            #hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        return forward

    def register_recr(net_, count, layer_name):
        if net_.__class__.__name__ == 'CrossAttention':
            if 'down_blocks_0_attentions_0' in layer_name:
                layer_name = 'down_blocks_0_attentions_0'
            elif 'down_blocks_0_attentions_1' in layer_name:
                layer_name = 'down_blocks_0_attentions_1'
            elif 'down_blocks_1_attentions_0' in layer_name:
                layer_name = 'down_blocks_1_attentions_0'
            elif 'down_blocks_1_attentions_1' in layer_name:
                layer_name = 'down_blocks_1_attentions_1'
            elif 'down_blocks_2_attentions_0' in layer_name:
                layer_name = 'down_blocks_2_attentions_0'
            elif 'down_blocks_2_attentions_1' in layer_name:
                layer_name = 'down_blocks_2_attentions_1'
            elif 'mid_block' in layer_name:
                layer_name = 'mid_block'

            elif 'up_blocks_1_attentions_0' in layer_name:
                layer_name = 'up_blocks_1_attentions_0'
            elif 'up_blocks_1_attentions_1' in layer_name:
                layer_name = 'up_blocks_1_attentions_1'
            elif 'up_blocks_1_attentions_2' in layer_name:
                layer_name = 'up_blocks_1_attentions_2'

            elif 'up_blocks_2_attentions_0' in layer_name:
                layer_name = 'up_blocks_2_attentions_0'
            elif 'up_blocks_2_attentions_1' in layer_name:
                layer_name = 'up_blocks_2_attentions_1'
            elif 'up_blocks_2_attentions_2' in layer_name:
                layer_name = 'up_blocks_2_attentions_2'

            elif 'up_blocks_3_attentions_0' in layer_name:
                layer_name = 'up_blocks_3_attentions_0'
            elif 'up_blocks_3_attentions_1' in layer_name:
                layer_name = 'up_blocks_3_attentions_1'
            elif 'up_blocks_3_attentions_2' in layer_name:
                layer_name = 'up_blocks_3_attentions_2'
            net_.forward = ca_forward(net_, layer_name)
            return count + 1
        elif hasattr(net_, 'children'):
            for name__, net__ in net_.named_children():
                full_name = f'{layer_name}_{name__}'
                count = register_recr(net__, count, full_name)
        return count

    cross_att_count = 0
    for net in unet.named_children():
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, net[0])
    controller.num_att_layers = cross_att_count
