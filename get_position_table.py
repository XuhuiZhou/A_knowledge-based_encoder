def get_position_table(self, words, position_dict, boundary = 2.2):
    words = words.cpu()
    words = words.data.numpy()
    switch_table = []
    for i in words:
        #s_dict = {}
        switch_line = []
        for index,j in enumerate(i):
            old_dict = position_dict.get(str(j))
            if old_dict is not None:
                keys = list(i)
                del keys[index-1:index+2]
                s_dict = {str(w): old_dict.get(str(w)) for w in keys}
                if j!=1 and j!=2:
                    #print(s_dict.keys())
                    max_rel = max(s_dict, key = s_dict.get)
                    value = s_dict[max_rel]
                    position = list(i).index(int(max_rel))+1
                    if value >= boundary:
                        switch_line.append(position)
                    else:
                        switch_line.append(0)
                else:
                    switch_line.append(0)
            else:
                switch_line.append(0)
        switch_table.append(switch_line)
    switch_table = np.array(switch_table)
    return torch.from_numpy(switch_table)