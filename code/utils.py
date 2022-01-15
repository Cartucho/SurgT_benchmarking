import os
import yaml
import dload


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


def download_case(case_name, case_data, case_path):
    for video_sample_number, video_sample_data in case_data.items():
        sample_path = os.path.join(case_path, video_sample_number)
        print("\t{}".format(sample_path))
        make_dir_if_needed(sample_path)
        for file_name, file_link in video_sample_data.items():
            file_path = os.path.join(sample_path, file_name)
            if os.path.isfile(file_path):
                print("\t\t{}: CHECK".format(file_path))
            else:
                print("\t\t{}: DOWNLOADING...".format(file_path))
                dload.save(file_link, file_path)


def download_folder(config_data):
    if config_data["is_to_download"]:
        path = os.path.join(config_data["dir"], config_data["subdir"])
        print("Checking download data in: {}".format(path))
        for case_name, case_data in config_data["cases"].items():
            case_path = os.path.join(path, case_name)
            download_case(case_name, case_data, case_path)


def download_data(config):
    validation = config["validation"]
    test = config["test"]
    download_folder(validation)
    download_folder(test)
