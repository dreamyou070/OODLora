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
        self.key_value_states_dict = {}
        self.repeat = 0
        self.normal_score_list = []
    def get_empty_store(self):
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def save_key_value_states(self, key_value_states, layer_name):
        if layer_name not in self.key_value_states_dict.keys() :
            self.key_value_states_dict[layer_name] = []
            self.key_value_states_dict[layer_name].append(key_value_states)
        else :
            self.key_value_states_dict[layer_name].append(key_value_states)
        return self.key_value_states_dict[layer_name].append(key_value_states)

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


    def cross_key_caching(self, key_value, layer_name):
        if layer_name not in self.cross_key_store.keys() :
            self.cross_key_store[layer_name] = []
            self.cross_key_store[layer_name].append(key_value)
        else :
            self.cross_key_store[layer_name].append(key_value)
        return key_value

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

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
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
        self.key_value_states_dict = {}
        self.repeat = 0
        self.normal_score_list = []