import os, sys
import imageio

def create_gif(image_path, model_name, save_path=None, duration=0.2):
    images = []
    file_names = sorted((image_path + "/"+fn for fn in os.listdir(image_path) if fn.endswith('.png')))
    for filename in file_names:
        images.append(imageio.imread(filename))
    if save_path is not None:
        output_file = save_path+'%s-GIF.gif' % model_name
    else:
        output_file = '%s-GIF.gif' % model_name
    imageio.mimsave(output_file, images, duration=duration)

# if __name__ == "__main__":
#     script = sys.argv.pop(0)
#     pth = '/home/antixk/Anand/Research/Code/Variational_optimizer/log/SNGAN/results/'
#     create_gif(pth, "SNGAN_Adam")
