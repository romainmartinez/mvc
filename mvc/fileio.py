import os
import numpy as np
import scipy.io as sio


class ImportMat:
    def __init__(self, data_path, data_format, export='mat'):
        if not os.path.isdir(data_path):
            raise ValueError('please provide a valid data path')
        self.FIELDS = ('datasets', 'participants', 'muscles', 'tests', 'relative_mvc')
        self.export = export
        self.data_path = data_path
        self.data_format = data_format
        self.datasets = []
        self.data, self.n = self._load_data()

    def _load_data(self):
        mat = {}
        count = 0
        print(f'data format: {self.data_format}')
        for ifile in os.listdir(self.data_path):
            if ifile.endswith(f'{self.data_format}.mat'):
                mat, n = self._import_mat_file(ifile, mat)
                count += n
        print('\ttotal participants: {}'.format(count))
        print(f'\tsample shape: {list(mat.values())[0].shape}')

        if self.export == 'mat':
            output = mat
        elif self.export == 'dict':
            output = self._to_dict(mat)
        elif self.export == 'array':
            output = self._to_array(mat)
        else:
            raise ValueError('please provide a valid export format (mat, dict or array)')

        return output, count

    def _import_mat_file(self, ifile, mat):
        name = ifile.replace('MVE_Data_', '').replace('.mat', '')
        self.datasets.append(name)
        path = os.path.join(self.data_path, ifile)
        mat[name] = sio.loadmat(path)['MVE']
        nb_participant = mat[name].shape[0]
        print("project '{}' loaded ({} participants)".format(name, nb_participant))
        return mat, nb_participant

    def _to_dict(self, mat):
        lists = {key: [] for key in self.FIELDS}
        for idataset, dataset_name in enumerate(list(mat.keys())):
            for iparticipant in range(mat[dataset_name].shape[0]):
                for imuscle in range(mat[dataset_name].shape[1]):
                    max_mvc = np.nanmax(mat[dataset_name][iparticipant, imuscle, :])
                    for itest in range(mat[dataset_name].shape[2]):
                        lists['participants'].append(iparticipant)
                        lists['datasets'].append(idataset)
                        lists['muscles'].append(imuscle)
                        lists['tests'].append(itest)
                        # normalize mvc (relative to max)
                        lists['relative_mvc'].append(
                            mat[dataset_name][iparticipant, imuscle, itest] * 100 / max_mvc
                        )
        return lists

    def _to_array(self, mat):
        lists = self._to_dict(mat)
        array = np.array([lists[self.FIELDS[0]], lists[self.FIELDS[1]], lists[self.FIELDS[2]],
                          lists[self.FIELDS[3]], lists[self.FIELDS[0]]]).T
        print(f'\t\toutput data shape: {array.shape}')
        return array


if __name__ == '__main__':
    path2data = '/media/romain/E/Projet_MVC/data/Final_output'
    dummy = ImportMat(path2data, data_format='only_max')
    print('')
