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


def get_case_paths_and_links(config_data):
    path = os.path.join(config_data["dir"], config_data["subdir"])
    case_paths = []
    case_links = []
    # Go through each of the cases
    for case_name, case_data in config_data["cases"].items():
        path_case = os.path.join(path, case_name)
        # Go through each of the samples of each case
        for sample_number, sample_links in case_data.items():
            path_case_sample = os.path.join(path_case, sample_number)
            case_paths.append(path_case_sample)
            case_links.append(sample_links)
    return case_paths, case_links


def download_case(case_paths, case_links):
    for case_sample_path, case_sample_links in zip(case_paths, case_links):
        print("\t{}".format(case_sample_path))
        make_dir_if_needed(case_sample_path)
        for file_name, file_link in case_sample_links.items():
            file_path = os.path.join(case_sample_path, file_name)
            if os.path.isfile(file_path):
                print("\t\t{}: CHECK".format(file_path))
            else:
                print("\t\t{}: DOWNLOADING...".format(file_path))
                dload.save(file_link, file_path)


def download_folder(config_data):
    if config_data["is_to_download"]:
        print("Checking if `{}` data is downloaded".format(config_data["subdir"]))
        case_paths, case_links = get_case_paths_and_links(config_data)
        download_case(case_paths, case_links)


def download_data(config):
    validation = config["validation"]
    test = config["test"]
    download_folder(validation)
    download_folder(test)
