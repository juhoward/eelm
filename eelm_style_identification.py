# deep learning utils
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
# standard utils
import os
import numpy as np
import random
import pickle
import argparse
# plotting utils
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
# clustering utilities
from umap import UMAP
from sklearn.cluster import DBSCAN
# custom utils 
from feature_extractor import build_model


# build dataset
# data_dir = '/home/jhoward/datasets/FLIR_Thermal/FLIR_ADAS_1_3/train/thermal_8_bit'
# data_dir = '/home/jhoward/datasets/FLIR_Thermal/FLIR_ADAS_1_3/train/RGB'
data_dir = '/opt/proj/arm-005II/datasets/KAIST-rgbt-ped-detection/data/kaist-rgbt/lwir'
# data_dir = '/opt/proj/arm-005II/datasets/KAIST-rgbt-ped-detection/data/kaist-rgbt/visible'

# setting random state for reproducibility
random_state = 42

IMG_SIZE = (640, 512)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('datatype')
parser.add_argument('-lf', '--load_features', action='store_true')
parser.add_argument('-lu', '--load_umap', action='store_true')
args = parser.parse_args()

def main():

    os.makedirs('output', exist_ok=True)

    if args.load_features:
        #  load saved features
        print('Loading saved image features...')
        with open('./output/' + args.datatype + '_feature_set.pkl', 'rb') as d:
            data = pickle.load(d)
    else:
        print('Extracting image features from ', data_dir)
        training = image_dataset_from_directory(
                data_dir,
                image_size=IMG_SIZE,
                shuffle=False,
                label_mode=None,
                batch_size=256
        )
        #  enable multi-gpu processing
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            # 1280 represents the final feature vector length of EfficientNetB0
            extractor = build_model(1280)

            outputs=[]
            for img_batch in iter(training):
                output = extractor(img_batch)
                outputs.append(np.squeeze(output.numpy()))
                print(outputs[-1].shape)
        # stack the outputs
        data = np.vstack(outputs)
        print('Final array shape: ', data.shape)
        print('Pickling Feature Set...')
        pickle.dump(data, open('./output/' + args.datatype + '_feature_set.pkl', 'wb'))


    if args.load_umap:
        print('Loading saved UMAP embedding...')
        with open('./output/' + args.datatype + '_UMAP_embedding.pkl', 'rb') as umap:
            X_embed = pickle.load(umap)
    else:
        # UMAP embedding
        print('Embedding data...')
        reducer = UMAP(random_state=random_state)
        X_embed = reducer.fit_transform(data)
        with open('./output/' + args.datatype + '_UMAP_embedding.pkl', 'wb') as f:
            pickle.dump(X_embed, f)
        # pickle.dump(X_embed, open('UMAP_embedding.pkl', 'wb'))
        print('Embedding saved to ', os.getcwd(), args.datatype + '_UMAP_embedding.pkl')
        print('Clustering on 2 dimensional UMAP embedding')

    # cluster assignment
    ########################### FLIR ADAS ###########################
    # dbscan2 = DBSCAN(eps = .7, min_samples = 30, algorithm = 'kd_tree', n_jobs = -1).fit(X_embed)

    ########################### KAIST visible #######################
    dbscan2 = DBSCAN(eps = .325, min_samples = 15, algorithm = 'kd_tree', n_jobs = -1).fit(X_embed)

    # plotting cluster labels
    clusters={}
    for label, point in zip(dbscan2.labels_, X_embed):
        if label not in clusters:
            clusters[label]=[]
        clusters[label].append(point)
#########################
        line = str(point) + ',' + str(label) + '\n'
        with open('./output/' + args.datatype + '_embedding_point_labels.txt', 'a') as ordered_embedding:
            ordered_embedding.write(line)
#########################
    vmax = len(clusters.keys()) - 2
    cNorm = colors.Normalize(vmin=-1, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='gist_ncar')
    plt.figure(figsize=(20,20))
    print('Found ', len(clusters.keys()), ' clusters. \n')
    for label in clusters:
        cluster=clusters[label]
        cluster=np.array(cluster)
        X=cluster[:,0]
        Y=cluster[:,1]
        plt.scatter(X,Y, s=15, color=scalarMap.to_rgba(label))

    plt.legend(clusters.keys(), bbox_to_anchor=(1, 1)) 
    plt.savefig('./output/' + args.datatype + '_clusters.png')
    plt.show()

    #  plotting cluster membership
    membership=dict()
    for key in clusters.keys():
        membership[key] = len(clusters[key])
    membership = dict(sorted(membership.items(), key=lambda item: item[1], reverse=True))
    labs = [str(i) for i in membership.keys()]
    cmap = [scalarMap.to_rgba(i) for i in membership.keys()]
    plt.figure(figsize=(10,5))
    plt.bar(labs, membership.values(), color=cmap)
    plt.xticks(labs)
    plt.title(args.datatype + ' Image Cluster Membership')
    plt.savefig('./output/' + args.datatype + '_cluster_membership.png')
    plt.show()

    # randomly sampling and plotting images from clusters
    print('Mapping images to DBSCAN labels...')
    DBLabels =  set(dbscan2.labels_)
    images = {k:[] for k in DBLabels}
    for label, image in zip(dbscan2.labels_, sorted(os.listdir(data_dir))):
        images[label].append(image)
    
    # writing labels file
    print('Writing labels to ', './output/' + args.datatype + '_cluster_labels.txt')
    with open('./output/' + args.datatype + '_cluster_labels.txt', 'a') as f:
        for label in images.keys():
            for idx in range(len(images[label])):
                line = images[label][idx] + ',' + str(label) + '\n'
                f.write(line)

    # making qualitative cluster evaluation graphic
    print('Randomly sampling each label...')
    examples = dict.fromkeys(clusters.keys())
    for label in DBLabels:
        examples[label]=[]
        attempt = 0
        while(len(examples[label]) < 3):
            attempt += 1
            idx = random.randint(0, len(images[label])-1)
            val = images[label][idx]
            if val not in examples[label]:
                examples[label].append(val)
            if attempt > 3:
                break
    print('Generating Qualitative evaluation plot...')
    fig, axs = plt.subplots(len(clusters.keys()), 3, figsize=(20,60))
    for r, label in enumerate(DBLabels):
        for c, img in enumerate(examples[label]):
            dir_img = Image.open(data_dir + '/' + img).convert('RGB')
            axs[(r,c)].imshow(dir_img)
            axs[(r,c)].set_title(f"Cluster {label}")
            axs[(r,c)].axis('off')
            

    plt.savefig('./output/' + args.datatype + '_examples.png')
    plt.show()

if __name__ == '__main__':
    main()