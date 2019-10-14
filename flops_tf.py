import tensorflow as tf
import argparse
import cv2

def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def main(pb_path,img):
    
    g2 = load_pb(pb_path)
    with g2.as_default():
        flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())
        print('FLOP after freezing', flops.total_float_ops)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_path','-p', dest='pb_path', required=True, help='REQUIRED: The path to pb')
    parser.add_argument('--img','-i', dest='input_image', required=True, help='REQUIRED: The input image')
    args = parser.parse_args()
    main(args.pb_path,args.input_image)
