import os
import scipy.io as sio
import numpy as np  # todo: delete this


class ImportMat:
    def __init__(self, datapath):
        if not os.path.isdir(datapath):
            raise ValueError('please provide a valid data path')
        self.datapath = datapath
        self.data, self.n = self.load_data()

    def load_data(self):
        mat = {}
        count = 0
        for ifile in os.listdir(self.datapath):
            if ifile.endswith('.mat'):
                mat, n = self.import_mat_file(ifile, mat)
                count += n
        print('\ttotal participants: {}'.format(count))
        return mat, count

    def import_mat_file(self, ifile, mat):
        name = ifile.replace('MVE_Data_', '').replace('.mat', '')
        path = os.path.join(self.datapath, ifile)
        mat[name] = sio.loadmat(path)['MVE']
        nparticipant = mat[name].shape[0]
        print("project '{}' loaded ({} participants)".format(name, nparticipant))
        return mat, nparticipant

    def mat2txt(data):

        with open('test.txt', 'wb') as outfile:
            shape = '# Array shape: {0}\n'.format(data.shape)
            outfile.write(bytes(shape, 'utf-8'))
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')
                outfile.write(b'# New slice\n')


if __name__ == '__main__':
    path2data = '/media/romain/E/Projet_MVC/data/Final_output'
    dummy = ImportMat(path2data)
    print()
