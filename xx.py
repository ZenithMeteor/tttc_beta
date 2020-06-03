file = open("./123.txt", "r")
configs = file.readlines()
file.close()
        
ip_dic={}
ip_dic_num=0
ip_list=[]
for config in configs:
    line = config.strip().split(' ')
    if ip_dic.get(line[0], 233) == 233:        
        _ = ip_dic.setdefault(line[0], ip_dic_num)
        ip_dic_num += 1
        #print([[line[1], line[2]]])
        ip_list.append([[line[1], line[2]]])
    else:
        ip_list[ip_dic[line[0]]].append([line[1], line[2]])
        
print(ip_dic)
print(ip_list)

for i in ip_dic:
    for f in ip_list[ip_dic[i]]:
        print(f)
