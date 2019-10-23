import os
import urllib.request
import argparse
import sys
import alexnet
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes

'''add some argument'''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Classify some images.',
                                 epilog= "And that's the category")
parser.add_argument('-m', '--mode', choices=['folder', 'url'], default='folder')
parser.add_argument('-p', '--path', help='Specify a path [e.g. testModel]', default = 'testModel')
args = parser.parse_args(sys.argv[1:])

if args.mode == 'folder':
    '''get image from folder'''
    withPath = lambda f: '{}/{}'.format(args.path,f)
    testImg = dict((f,cv2.imread(withPath(f))) for f in os.listdir(args.path) if os.path.isfile(withPath(f)))
elif args.mode == 'url':
    def url2img(url):
        '''get image from url'''
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    testImg = {args.path:url2img(args.path)}

if testImg.values():
    #define some alexNet's params
    dropoutPro = 1
    classNum = 1000
    skip = []

    imgMean = np.array([104, 117, 124], np.float)
    x = tf.placeholder("float", [1, 227, 227, 3])

    model = alexnet.alexNet(x, dropoutPro, classNum, skip)
    score = model.fc3
    softmax = tf.nn.softmax(score)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)

        for key,img in testImg.items():
            '''img preprocess'''
            resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean
            maxx = np.argmax(sess.run(softmax, feed_dict = {x: resized.reshape((1, 227, 227, 3))}))
            res = caffe_classes.class_names[maxx]

            print("{}: {}\n----".format(key,res))
            cv2.imshow("{}".format(res), img)
            cv2.waitKey(0)
