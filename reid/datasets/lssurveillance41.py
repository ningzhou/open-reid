from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

class LsSurveillance41(Dataset):
    def __init__(self, root, split_id=0, num_val=0.3, download=False, orig_uri=None, max_imgs_percam=10):
        super(LsSurveillance41, self).__init__(root, split_id=split_id)

        if download:
            self.random_select(orig_uri, max_imgs_percam)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download/generate it.")

        self.load(num_val)

    def random_select(self, orig_uri, max_imgs_percam):
        import re
        import os
        import glob
        import shutil

        if self._check_integrity():
            print("Files already generated and verified")
            return

        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)
        # 41 identities with 42 camera views each
        identities = [[[] for _ in range(42)] for _ in range(41)]

        def register(subdir, max_imgs_percam):
            pid = int(osp.basename(subdir)[0:4]) - 1
            assert 0 <= pid < 41
            cam_dirs = sorted(os.listdir(subdir))
            pattern = re.compile(r'cam(\d+)')
            for camdir in cam_dirs:
                camid = int(pattern.search(camdir).groups()[0])
                assert 1 <= camid <=42
                camid -= 1
                fpaths = sorted(glob.glob(osp.join(subdir, camdir, '*.jpg')))
                if len(fpaths) > max_imgs_percam:
                    rand_indices = np.random.permutation(len(fpaths)).tolist()
                    rand_indices = rand_indices[0:max_imgs_percam]
                    subfpaths = [fpaths[i] for i in rand_indices]
                    fpaths = subfpaths

                for fpath in fpaths:
                    fname = ('{:08d}_{:02d}_{:04d}.jpg'
                             .format(pid, camid, len(identities[pid][camid])))
                    identities[pid][camid].append(fname)
                    shutil.copy(fpath, osp.join(images_dir, fname))

        pid_dirs = sorted(os.listdir(orig_uri))
        for subdir in pid_dirs:
            register(osp.join(orig_uri, subdir), max_imgs_percam)

        # save meta information into a json file
        meta = {'name': 'LsSurveillance41',
                'shot': 'multiple',
                'num_cameras': 42,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        num = len(identities)
        splits = []
        # Put all ids into query and gallery
        pids = np.random.permutation(num).tolist()
        trainval_pids = sorted(pids[:num // 2])
        test_pids = sorted(pids[num // 2:])
        split = {'trainval': trainval_pids,
                 'query': pids,
                 'gallery': pids}
        splits.append(split)
        # Randomly create training and test splits
        for _ in range(10):
            pids = np.random.permutation(num).tolist()
            trainval_pids = sorted(pids[:num // 2])
            test_pids = sorted(pids[num // 2:])
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)

        write_json(splits, osp.join(self.root, 'splits.json'))








