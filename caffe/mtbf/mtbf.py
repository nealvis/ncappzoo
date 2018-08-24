#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IEPlugin


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=False, default="./bvlc_googlenet.xml", type=str)
    parser.add_argument("-i", "--input", help="Path to a text file with image file names and true top 1 results. Each line has a full path to image file, a tab, and the true top 1 result.  An inference will be executed for each file and top 1 compared with the true value from this file.", required=False, default="./val_top1_true.txt",
                        type=str)
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (MYRIAD by default)", default="MYRIAD",
                        type=str)
    #parser.add_argument("-nt", "--number_top", help="Number of top results", default=10, type=int)

    return parser


def main():

    args = build_argparser()
    try:
        args.parse_args()
    except:
        args.print_help()
        raise

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)
    # Read IR
    net = IENetwork.from_ir(model=model_xml, weights=model_bin)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob]

    # Load network to the plugin
    exec_net = plugin.load(network=net)

    wrong_counter = 0
    right_counter = 0

    print ("model is: " + args.model)
    print ("input is: " + args.input)
    input_file = open(args.input, "r")
    lines = input_file.readlines()
    line_number = 0;
    for a_line in lines:
        line_number += 1
        if (a_line == None or len(a_line) < 3):
            break
        my_list = a_line.split("\t");
        if (len(my_list) != 2):
            break
        image_filename = my_list[0]
        try:
            image_top1_expected = int(my_list[1])
            if (image_top1_expected < 0):
                raise Exception("Negative top 1")
        except:
            print("Error (line: + " + str(line_number) + ") " + "bad top 1 value")
            continue

        if (not os.path.isfile(image_filename)):
            image_filename = os.path.split(image_filename)[1]

        #print ("image file name: " + image_filename, "   expected top 1: " + str(image_top1_expected))

        image = cv2.imread(image_filename)
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        image = image.reshape((n, c, h, w))

        # Start sync inference
        res = exec_net.infer(inputs={input_blob: image})

        top_ind = np.argsort(res[out_blob], axis=1)[0, -1:][::-1]
        image_top1_actual = top_ind[0]
        #print(image_filename + " expected: " + str(image_top1_expected) + "  actual: " + str(image_top1_actual) + " probability: " + str(res[out_blob][0, image_top1_actual]))
        if (image_top1_expected != image_top1_actual):
            wrong_counter += 1
            print ("Wrong Result: " + image_filename + " : Expected: " + str(image_top1_expected) + "   Actual: " + str(image_top1_actual) + " probability: " + str(res[out_blob][0,image_top1_actual]))
        else:
            right_counter += 1
        #for i in top_ind:
        #    print("%f #%d" % (res[out_blob][0, i], i))

    input_file.close()
    del net

    print("")
    print("--------------------------------------------------------------")
    print("mtbf complete for " + str(wrong_counter + right_counter) + " images.")
    print("  Correct inferences: " + str(right_counter))
    print("  Incorrect inferences: " + str(wrong_counter))

    if (wrong_counter == 0):
        print("All images passed.")
    print("--------------------------------------------------------------")

#
#    input_file = args.input
#
#    image = cv2.imread(args.input)
#    image = cv2.resize(image, (w, h))
#    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
#    image = image.reshape((n, c, h, w))
#    # Load network to the plugin
#    exec_net = plugin.load(network=net)
#    del net
#    # Start sync inference
#    res = exec_net.infer(inputs={input_blob: image})
#    top_ind = np.argsort(res[out_blob], axis=1)[0, -args.number_top:][::-1]
#    for i in top_ind:
#        print("%f #%d" % (res[out_blob][0, i], i))
#    del exec_net
#    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)
