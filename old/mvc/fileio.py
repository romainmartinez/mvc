import numpy as np
import scipy.io as sio


def load_data(data_path, data_format, normalize=False, verbose=False):
    mat = {}
    data = {key: [] for key in ('datasets', 'participants', 'muscles', 'tests', 'mvc')}
    count = -1
    dataset_names = []

    for idataset, ifile in enumerate(data_path.iterdir()):
        if ifile.parts[-1].endswith(f'{data_format}.mat'):
            dataset = ifile.parts[-1].replace('_only_max.mat', '').replace('MVE_Data_', '')

            if dataset not in dataset_names:
                dataset_names.append(dataset)

            mat[dataset] = sio.loadmat(ifile)['MVE']
            n_participants = mat[dataset].shape[0]
            if verbose:
                print(f"project '{dataset}' ({n_participants} participants)")

            for iparticipant in range(mat[dataset].shape[0]):
                count += 1
                for imuscle in range(mat[dataset].shape[1]):
                    max_mvc = np.nanmax(mat[dataset][iparticipant, imuscle, :])
                    for itest in range(mat[dataset].shape[2]):
                        data['participants'].append(count)
                        data['datasets'].append(idataset)
                        data['muscles'].append(imuscle)
                        data['tests'].append(itest)
                        if normalize:
                            data['mvc'].append(mat[dataset][iparticipant, imuscle, itest] * 100 / max_mvc)
                        else:
                            data['mvc'].append(mat[dataset][iparticipant, imuscle, itest])

    if verbose:
        print(f'\n\ttotal participants: {count}')
    return data, dataset_names
