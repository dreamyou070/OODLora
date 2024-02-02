import abc
class AttentionStore :

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.heatmap_store = {}
        self.self_query_store = {}
        self.self_key_store = {}
        self.self_value_store = {}
        self.cross_query_store = {}
        self.cross_key_store = {}
        self.cross_value_store = {}
        self.query_dict = {}
        self.key_dict = {}
        self.value_dict = {}
        self.repeat = 0
        self.normal_score_list = []
    def get_empty_store(self):
        return {}

    def save_query(self, query, layer_name):
        if layer_name not in self.query_dict.keys():
            self.query_dict[layer_name] = []
            self.query_dict[layer_name].append(query)
        else:
            self.query_dict[layer_name].append(query)

    def save_key(self, key, layer_name):
        if layer_name not in self.key_dict.keys():
            self.key_dict[layer_name] = []
            self.key_dict[layer_name].append(key)
        else:
            self.key_dict[layer_name].append(key)


    def store_normal_score(self, score):
        self.normal_score_list.append(score)

    def store(self, attn, layer_name):
        if layer_name not in self.step_store.keys() :
            self.step_store[layer_name] = []
            self.step_store[layer_name].append(attn)
        else :
            self.step_store[layer_name].append(attn)
            #self.step_store[layer_name] = self.step_store[layer_name] + attn
        return attn


    def self_query_key_value_caching(self,query_value, key_value, value_value, layer_name):

        if layer_name not in self.self_query_store.keys() :
            self.self_query_store[layer_name] = []
            self.self_key_store[layer_name] = []
            self.self_value_store[layer_name] = []
            self.self_query_store[layer_name].append(query_value)
            self.self_key_store[layer_name].append(key_value)
            self.self_value_store[layer_name].append(value_value)
        else :
            self.self_query_store[layer_name].append(query_value)
            self.self_key_store[layer_name].append(key_value)
            self.self_value_store[layer_name].append(value_value)

    def save(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        self.step_store[key].append(attn.clone())
        return attn

    def between_steps(self):
        assert len(self.attention_store) == 0

        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.heatmap_store = {}
        self.self_query_store = {}
        self.self_key_store = {}
        self.self_value_store = {}
        self.cross_query_store = {}
        self.cross_key_store = {}
        self.cross_value_store = {}
        self.query_dict = {}
        self.key_dict = {}
        self.value_dict = {}
        self.repeat = 0
        self.normal_score_list = []