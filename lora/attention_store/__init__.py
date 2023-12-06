import abc
class AttentionStore :

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.heatmap_store = {}
    def get_empty_store(self):
        return {}

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

    def self_query_key_caching(self,query_value, key_value, layer_name):
        if layer_name not in self.self_query_store.keys() :
            self.self_query_store[layer_name] = []
            self.self_key_store[layer_name] = []
            self.self_query_store[layer_name].append(query_value)
            self.self_key_store[layer_name].append(key_value)
        else :
            self.self_query_store[layer_name].append(query_value)
            self.self_key_store[layer_name].append(key_value)
        return query_value, key_value

    def reset(self):
        self.step_store = {}
        self.attention_store = {}
