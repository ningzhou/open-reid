from unittest import TestCase

class TestLsSurveillance41(TestCase):
    def test_init(self):
        import os.path as osp
        from reid.datasets.lssurveillance41 import LsSurveillance41
        from reid.utils.serialization import read_json
        from reid.utils.osutils import mkdir_if_missing

        raw_dir = '/home/niz/data/ls-surveillance/singleperson41/singleperson41_acf'
        root_dir = '/home/niz/src/open-reid/examples/data/singleperson41_acf'
        mkdir_if_missing(root_dir)

        root, split_id, num_val = root_dir, 0, 10
        dataset = LsSurveillance41(root, split_id, num_val=num_val, download=True,
                                   orig_uri=raw_dir, max_imgs_percam=10)

        self.assertTrue(osp.isfile(osp.join(root, 'meta.json')))
        self.assertTrue(osp.isfile(osp.join(root, 'splits.json')))
        meta = read_json(osp.join(root, 'meta.json'))
        self.assertEquals(len(meta['identities']), 41)
        splits = read_json(osp.join(root, 'splits.json'))
        self.assertEquals(len(splits), 11)

        self.assertDictEqual(meta, dataset.meta)
        self.assertDictEqual(splits[split_id], dataset.split)