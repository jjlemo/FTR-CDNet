class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"

    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = '/data0/mylevir/dataset/LEVIR'
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir = '/data0/mylevir/dataset/whu'
        elif data_name == 'S2Looking':
            self.label_transform = "norm"
            self.root_dir = '/data0/mylevir/dataset/S2Looking'
        elif data_name == 'WHU_CD':
            self.label_transform = "norm"
            self.root_dir = r'/data0/mylevir/dataset/WHU_CD_256'
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        elif data_name == 'quick_start_SECOND':
            self.root_dir = 'F:\\little\\dataset\\SECOND\\'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)
