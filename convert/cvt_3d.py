import torch

path = '/data-input/my_CLOCs/log/second/train-3-no/result/27.pt'
data = torch.load(path)
save_path = '/data-input/data/clocs_data/result/second_cascade_o'

for i in range(len(data)):
    name = data[i]['frame_id'] + '.txt'
    f = open(save_path + '/' + name,'w')
    for j in range(len(data[i]['bbox'])):
        bbox = "{:.2f} {:.2f} {:.2f} {:.2f}".format(data[i]['bbox'][j][0],data[i]['bbox'][j][1],data[i]['bbox'][j][2],data[i]['bbox'][j][3])
        to_write = ""
        to_write += str(data[i]['name'][j])+' '
        to_write += str(data[i]['truncated'][j]) +' '
        to_write += str(data[i]['occluded'][j])+' '
        to_write += str("{:.2f}".format(data[i]['alpha'][j]))+' '
        to_write += bbox + ' '
        to_write += "{:.2f} {:.2f} {:.2f}".format(data[i]['dimensions'][j][1],data[i]['dimensions'][j][2],data[i]['dimensions'][j][0]) +' '
        to_write += "{:.2f} {:.2f} {:.2f}".format(data[i]['location'][j][0],data[i]['location'][j][1],data[i]['location'][j][2]) + ' '
        to_write += str("{:.2f}".format(data[i]['rotation_y'][j]))+' '
        to_write += str("{:.2f}".format(data[i]['score'][j]))
        f.write(to_write+'\n')
    f.close()

print(1)