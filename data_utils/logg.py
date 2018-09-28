import os
import shutil

def get_save_path(model_name, overwrite=False):
    cwd = os.getcwd()
    folder_name =  os.path.basename(cwd)
    save_path = cwd[:-len(folder_name)] + 'log/'
    asset_path = cwd[:-len(folder_name)] + 'assets/'
    data_path = cwd[:-len(folder_name)] + 'datasets/'
    model_path = save_path + model_name
    save_model_path = model_path + "/saved_models"
    save_result_path = model_path + "/results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Delete Previous Model and its results
    if overwrite == True:
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        os.makedirs(model_path)
        os.makedirs(save_model_path)
        os.makedirs(save_result_path)

        # Create gitignore for the results
        with open(save_result_path+"/.gitignore", "w") as file:
            file.write("* \n !.gitignore")

    return save_model_path, save_result_path, asset_path, data_path

if __name__ == "__main__":
    get_save_path("sheep")