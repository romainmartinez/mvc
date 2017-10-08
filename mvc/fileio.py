import os
import scipy.io as sio


class ImportMat:
    def __init__(self, data_path):
        if not os.path.isdir(data_path):
            raise ValueError('please provide a valid data path')
        self.data_path = data_path
        self.data, self.n = self.load_data()

    def load_data(self):
        mat = {}
        count = 0
        for ifile in os.listdir(self.data_path):
            if ifile.endswith('.mat'):
                mat, n = self.import_mat_file(ifile, mat)
                count += n
        print('\ttotal participants: {}'.format(count))
        return mat, count

    def import_mat_file(self, ifile, mat):
        name = ifile.replace('MVE_Data_', '').replace('.mat', '')
        path = os.path.join(self.data_path, ifile)
        mat[name] = sio.loadmat(path)['MVE']
        nb_participant = mat[name].shape[0]
        print("project '{}' loaded ({} participants)".format(name, nb_participant))
        return mat, nb_participant

    @staticmethod
    def mat2txt(data):
        import numpy as np  # here because this method is never used
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
