import re

class utils:
    def read_component(comps):
        component = {}
        tokens = re.findall('[A-Z][a-z]*|\d+\.\d+|\d+', comps)
        i = 0
        while i < len(tokens):
            if(i+1 >= len(tokens) or tokens[i+1].isalpha()):
                component[tokens[i]] = 1.0
                i += 1
            else:
                component[tokens[i]] = float(tokens[i+1])
                i += 2
        return component

    def get_component_vector(atom_table, comps):
        component = utils.read_component(comps)
        vector = (max(atom_table.values())+1) * [0]
        sum_num = float(sum(list(component.values())))
        for atom in component.keys():
            vector[atom_table[atom]] = component[atom]/sum_num
        return vector

    def read_data(data_path, flag=0):
        compss = []
        Tc = []
        with open(data_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                token = line.split()
                compss.append(token[0])
                if flag == 0:
                    Tc.append(float(token[1]))
        if flag == 0:
            return compss, Tc
        return compss

    def embedding(component_vector):
        embed_vector = []
        for component in component_vector:
            embed = []
            for i in range(0, len(component)):
                if component[i] != 0:
                    embed.append([i+1, component[i]])
                if len(embed) >= 10:
                    break
            while len(embed) < 10:
                embed.append([-2**32, -2**32])
            embed_vector.append(embed)
        return embed_vector