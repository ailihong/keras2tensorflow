import tensorflow as tf
import argparse
import cv2

IMAGE_HEIGHT=64
IMAGE_WIDTH=64

def main(pb_path,img):
    image = cv2.imread(img)
    image = cv2.resize(image,(64,64))
    
    image_in = image.reshape(1, 64,64,3)
    
    with open(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        _input = tf.get_default_graph().get_tensor_by_name("input_1:0")
        _output = tf.get_default_graph().get_tensor_by_name("k2tfout_0:0")
        out = sess.run(_output,feed_dict={_input:image_in})
        print(out)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_path','-p', dest='pb_path', required=True, help='REQUIRED: The path to pb')
    parser.add_argument('--img','-i', dest='input_image', required=True, help='REQUIRED: The input image')
    args = parser.parse_args()
    main(args.pb_path,args.input_image)
