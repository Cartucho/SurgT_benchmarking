import os
import yaml
import dload


class CaseSample:
    def __init__(self, case_id, case_sample_path, case_sample_links):
        self.case_id = case_id
        self.case_sample_path = case_sample_path
        self.case_sample_links = case_sample_links

    def download_case_sample_data(self):
        print("\t{}".format(self.case_sample_path))
        make_dir_if_needed(self.case_sample_path)
        for file_name, file_link in self.case_sample_links.items():
            file_path = os.path.join(self.case_sample_path, file_name)
            if os.path.isfile(file_path):
                print("\t\t{}: CHECK".format(file_path))
            else:
                print("\t\t{}: DOWNLOADING...".format(file_path))
                dload.save(file_link, file_path)


def is_path_file(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), string)


def load_yaml_data(path):
    if is_path_file(path):
        with open(path) as f_tmp:
            return yaml.load(f_tmp, Loader=yaml.FullLoader)


def write_yaml_data(path, data):
    with open(path, 'w') as fp:
        yaml.dump(data, fp)


def make_dir_if_needed(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def get_case_samples(config_data):
    path = os.path.join(config_data["dir"], config_data["subdir"])
    case_samples = []
    # Go through each of the cases
    for case_id, case_data in config_data["cases"].items():
        path_case = os.path.join(path, case_id)
        # Go through each of the samples of each case
        for sample_number, sample_links in case_data.items():
            path_case_sample = os.path.join(path_case, sample_number)
            cs = CaseSample(case_id, path_case_sample, sample_links) # + 1 to start in case_1
            case_samples.append(cs)
    return case_samples


def download_folder(config_data):
    if config_data["is_to_download"]:
        print("DOWNLOAD: Checking `{}` data:".format(config_data["subdir"]))
        case_samples = get_case_samples(config_data)
        for cs in case_samples:
            cs.download_case_sample_data()


def download_data(config):
    validation = config["validation"]
    test = config["test"]
    download_folder(validation)
    download_folder(test)
