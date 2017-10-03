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


if __name__ == '__main__':
    path2data = '/media/romain/E/Projet_MVC/data/Final_output'
    dummy = ImportMat(path2data)
    print()

    for idataset, matrix in enumerate(dummy.data.values()):
        # preallocate
        participants, datasets, muscles, tests, relative_mvc = (np.array([], dtype='int') for i in range(5))
        absolute_mvc = np.array([], dtype='float')
        for (iparticipant, imuscle, itest), mvc in np.ndenumerate(matrix):
            participants = np.append(participants, iparticipant + 1)
            datasets = np.append(datasets, idataset + 1)
            muscles = np.append(muscles, imuscle + 1)
            tests = np.append(tests, itest + 1)
            absolute_mvc = np.append(absolute_mvc, mvc)
        print()
    print()
