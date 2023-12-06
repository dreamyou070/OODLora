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
    def get_empty_store(self):
        return {}

    def save_key_value_states(self, key_value_states, layer_name):
        if layer_name not in self.key_value_states_dict.keys() :
            self.key_value_states_dict[layer_name] = []
            self.key_value_states_dict[layer_name].append(key_value_states)
        else :
            self.key_value_states_dict[layer_name].append(key_value_states)
        return self.key_value_states_dict[layer_name].append(key_value_states)


    def store(self, attn, layer_name):
        if layer_name not in self.step_store.keys() :
            self.step_store[layer_name] = []
            self.step_store[layer_name].append(attn)
        else :
            self.step_store[layer_name].append(attn)
            #self.step_store[layer_name] = self.step_store[layer_name] + attn
        return attn

    def save(self, word_heat_map, layer_name):
        if layer_name not in self.heatmap_store.keys() :
            self.heatmap_store[layer_name] = []
            self.heatmap_store[layer_name].append(word_heat_map)
        else :
            self.heatmap_store[layer_name].append(word_heat_map)


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
        return query_value, key_value, value_value

    def cross_query_key_value_caching(self, query_value, key_value, value_value, layer_name):
        if layer_name not in self.cross_query_store.keys() :
            self.cross_query_store[layer_name] = []
            self.cross_key_store[layer_name] = []
            self.cross_value_store[layer_name] = []
            self.cross_query_store[layer_name].append(query_value)
            self.cross_key_store[layer_name].append(key_value)
            self.cross_value_store[layer_name].append(value_value)
        else :
            self.cross_query_store[layer_name].append(query_value)
            self.cross_key_store[layer_name].append(key_value)
            self.cross_value_store[layer_name].append(value_value)
        return query_value, key_value, value_value
    def cross_key_value_caching(self, key_value, value_value, layer_name):

        if layer_name not in self.cross_key_store.keys() :
            self.cross_key_store[layer_name] = []
            self.cross_value_store[layer_name] = []
            self.cross_key_store[layer_name].append(key_value)
            self.cross_value_store[layer_name].append(value_value)

        else :
            self.cross_key_store[layer_name].append(key_value)
            self.cross_value_store[layer_name].append(value_value)
        return key_value, value_value

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