"""
script used to make use of the entire Kohn-Canade dataset
"""

import os
from shutil import copy

from cnn.scripts.utils_datasets.common import dictionary


def run():
    dir = "dataset_cke_large"
    if not os.path.exists(dir):
        os.mkdir(dir)
    for root, dirs, files in os.walk("cohn-kanade\emotion_labels\Emotion", topdown=True):
        for file in files:
            name = file.split('_')
            print(name)
            emotion = open(os.path.join(root, file), "r").read()
            number = int(emotion.split('.')[0])
            dest = os.path.join(dir, dictionary[number])
            if not os.path.exists(dest):
                os.mkdir(dest)
            sources= "cohn-kanade/cohn-kanade-dataset/" + name[0] + "/" + name[1]
            for root_inner,dirs_inner,files_inner in os.walk(sources):
                for file_inner in files:
                    if file_inner == ".DS_Store":
                        continue
                    src =os.path.join(root_inner, file_inner)
                    print(file_inner)
                    file_number = int(file_inner.split('_')[-1][0:-4])
                    if file_number in [i for i in range(0,5)]:
                        neutral = os.path.join(dir, "neutral")
                        if not os.path.exists(neutral):
                            os.mkdir(neutral)
                        copy(src, neutral)
                    else:
                        print("copying from {} to {}".format(src,dest))
                    copy(src, dest)


if __name__ == "__main__":
    run()
