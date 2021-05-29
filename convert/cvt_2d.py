import pickle 
import mmcv

path = '/data-input/data/clocs_data/2D/pkl/cascade_10.pkl'
save_path = '/data-input/data/clocs_data/2D/cascade'
data = pickle.load(open(path,'rb'))
for i in range(len(data)):
    name = '{:0>6d}.txt'.format(i)
    f = open(save_path + '/' + name,'w')
    for j in range(len(data[i][0])):
        bbox = "{:.2f} {:.2f} {:.2f} {:.2f}".format(data[i][0][j][0],data[i][0][j][1],data[i][0][j][2],data[i][0][j][3])
        to_write = ""
        to_write += str('Car')+' '
        to_write += str('-1')+' '
        to_write += str('-1')+' '
        to_write += str('-10')+' '
        to_write += bbox + ' '
        to_write += '-1 -1 -1 '
        to_write += '-1 -1 -1 '
        to_write += '-10 '
        to_write += str(data[i][0][j][4])
        f.write(to_write+'\n')
    f.close()

print('done')