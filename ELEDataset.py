
import os.path as osp
import glob
import torchreid
from torchreid.data import ImageDataset

class NewDataset(ImageDataset):
    dataset_dir = 'embodied-learning-data-test'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        self.query_pids = set()
        train = self.process_dir(self.train_dir, mode='train')
        query = self.process_dir(self.query_dir, mode='query')
        gallery = self.process_dir(self.gallery_dir, mode='gallery')

        super(NewDataset, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, mode='train'):
        img_paths = glob.glob(dir_path+'/**/*.jpg', recursive=True)

        pid_container = set()

        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            if mode == 'train':
                pid_container.add(pid)
            elif mode == 'query':
                self.query_pids.add(pid)
        if mode == 'train':
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            pid = int(img_name.split('_')[0])
            if mode == 'gallery':

                if pid not in self.query_pids:
                    continue
                camid = 1
            else:
                camid = 0

            if mode == 'train':
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data