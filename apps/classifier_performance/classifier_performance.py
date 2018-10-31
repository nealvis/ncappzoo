#! /usr/bin/env python3

# Copyright(c) 2017-2018 Intel Corporation.
# License: MIT See LICENSE file in root directory.



from ubuntu16.openvino.inference_engine import IENetwork, IEPlugin, ExecutableNetwork, ie_api
import ubuntu16.openvino.inference_engine.ie_api

import cv2
import numpy
import time
import sys
import threading
import os
from sys import argv
import datetime
import queue

from queue import *

#set some global parameters to initial values that may get overriden with arguments to the appliation.
number_of_devices = 1
number_of_inferences = 200
run_async = True
time_threads = True
time_main = False
#async_algo = "complete"  # either "experiment" or "complete."
threads_per_dev = 2 # for each device one executable network will be created and this many threads will be
                    # created to run inferences in parallel on that executable network
simultaneous_infer_per_thread = 4  # Each thread will start this many async inferences at at time.
                                   # it should be at least the number of NCEs on board.  The Myriad X has 2
                                   # seem to get slightly better results more. Myriad X does well with 4
report_interval = 400 #report out the current FPS every this many inferences

sep = os.path.sep

model_xml_fullpath = "." + sep + "bvlc_googlenet.xml"
model_bin_fullpath = "." + sep + "bvlc_googlenet.bin"

#model_xml_fullpath = "." + sep + "googlenet_mx_val_40000.xml"
#model_bin_fullpath = "." + sep + "googlenet_mx_val_40000.bin"


# net_config = {"HW_STAGES_OPTIMIZATION": "YES"}
# , "COMPUTE_LAYOUT":"VPU_NHCW"}
#             "KEY_VPU_RESHAPE_OPTIMIZATION" : "NO"}
# plugin.set_config({"VPU_KEY_HW_STAGES_OPTIMIZATION": "YES"})
#              "KEY_VPU_RESHAPE_OPTIMIZATION" : "NO"})
net_config = {'HW_STAGES_OPTIMIZATION': 'YES', 'COMPUTE_LAYOUT':'VPU_NCHW', 'RESHAPE_OPTIMIZATION':'NO'}


def handle_args():
    """Reads the commandline args and adjusts initial values of globals values to match

    :return: False if there was an error with the args, or True if args processed ok.
    """
    global number_of_devices, number_of_inferences, model_xml_fullpath, model_bin_fullpath, run_async, \
           time_threads, time_main, num_ncs_devs, threads_per_dev, simultaneous_infer_per_thread, report_interval

    have_model_xml = False
    have_model_bin = False

    for an_arg in argv:
        lower_arg = str(an_arg).lower()
        if (an_arg == argv[0]):
            continue

        elif (lower_arg == 'help'):
            return False

        elif (lower_arg.startswith('num_devices=') or lower_arg.startswith("nd=")):
            try:
                arg, val = str(an_arg).split('=', 1)
                num_dev_str = val
                number_of_devices = int(num_dev_str)
                if (number_of_devices < 0):
                    print('Error - num_devices argument invalid.  It must be > 0')
                    return False
                print('setting num_devices: ' + str(number_of_devices))
            except:
                print('Error - num_devices argument invalid.  It must be between 1 and number of devices in system')
                return False;

        elif (lower_arg.startswith('report_interval=') or lower_arg.startswith("ri=")):
            try:
                arg, val = str(an_arg).split('=', 1)
                val_str = val
                report_interval = int(val_str)
                if (report_interval < 0):
                    print('Error - report_interval must be greater than or equal to 0')
                    return False
                print('setting report_interval: ' + str(report_interval))
            except:
                print('Error - report_interval argument invalid.  It must be greater than or equal to zero')
                return False;

        elif (lower_arg.startswith('num_inferences=') or lower_arg.startswith('ni=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                num_infer_str = val
                number_of_inferences = int(num_infer_str)
                if (number_of_inferences < 0):
                    print('Error - num_inferences argument invalid.  It must be > 0')
                    return False
                print('setting num_inferences: ' + str(number_of_inferences))
            except:
                print('Error - num_inferences argument invalid.  It must be between 1 and number of devices in system')
                return False;

        elif (lower_arg.startswith('num_threads_per_device=') or lower_arg.startswith('ntpd=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                val_str = val
                threads_per_dev = int(val_str)
                if (threads_per_dev < 0):
                    print('Error - threads_per_dev argument invalid.  It must be > 0')
                    return False
                print('setting num_threads_per_device: ' + str(threads_per_dev))
            except:
                print('Error - num_threads_per_device argument invalid, it must be a positive integer.')
                return False;

        elif (lower_arg.startswith('num_simultaneous_inferences_per_thread=') or lower_arg.startswith('nsipt=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                val_str = val
                simultaneous_infer_per_thread = int(val_str)
                if (simultaneous_infer_per_thread < 0):
                    print('Error - simultaneous_infer_per_thread argument invalid.  It must be > 0')
                    return False
                print('setting num_simultaneous_inferences_per_thread: ' + str(simultaneous_infer_per_thread))
            except:
                print('Error - num_simultaneous_inferences_per_thread argument invalid, it must be a positive integer.')
                return False;

        elif (lower_arg.startswith('model_xml=') or lower_arg.startswith('mx=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                model_xml_fullpath = val
                if not (os.path.isfile(model_xml_fullpath)):
                    print("Error - Model XML file passed does not exist or isn't a file")
                    return False
                print('setting model_xml: ' + str(model_xml_fullpath))
                have_model_xml = True
            except:
                print('Error with model_xml argument.  It must be a valid model file generated by the OpenVINO Model Optimizer')
                return False;

        elif (lower_arg.startswith('model_bin=') or lower_arg.startswith('mb=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                model_bin_fullpath = val
                if not (os.path.isfile(model_bin_fullpath)):
                    print("Error - Model bin file passed does not exist or isn't a file")
                    return False
                print('setting model_bin: ' + str(model_bin_fullpath))
                have_model_bin = True
            except:
                print('Error with model_bin argument.  It must be a valid model file generated by the OpenVINO Model Optimizer')
                return False;

        elif (lower_arg.startswith('run_async=') or lower_arg.startswith('ra=')) :
            try:
                arg, val = str(an_arg).split('=', 1)
                run_async = (val.lower() == 'true')
                print ('setting run_async: ' + str(run_async))
            except:
                print("Error with run_async argument.  It must be 'True' or 'False' ")
                return False;

        #elif (lower_arg.startswith('async_algo=') or lower_arg.startswith('aa=')) :
        #    try:
        #        arg, val = str(an_arg).split('=', 1)
        #        async_algo = val.lower()
        #        if (async_algo != "complete" and  async_algo != "experiment"):
        #            print("Error with async_algo argument.  It must be complete or experiment")
        #            return False
        #        print ('setting async_algo: ' + async_algo)
        #
        #    except:
        #        print("Error with run_async argument.  It must be 'True' or 'False' ")
        #        return False;

        elif (lower_arg.startswith('time_threads=') or lower_arg.startswith('tt=')) :
            try:
                arg, val = str(an_arg).split('=', 1)
                time_threads = (val.lower() == 'true')
                print ('setting time_threads: ' + str(time_threads))
            except:
                print("Error with time_threads argument.  It must be 'True' or 'False' ")
                return False;

        elif (lower_arg.startswith('time_main=') or lower_arg.startswith('tm=')) :
            try:
                arg, val = str(an_arg).split('=', 1)
                time_main = (val.lower() == 'true')
                print ('setting time_main: ' + str(time_main))
            except:
                print("Error with time_main argument.  It must be 'True' or 'False' ")
                return False;


    if (time_main == False and time_threads == False):
        print("Error - Both time_threads and time_main args were set to false.  One of these must be true. ")
        return False

    if ((have_model_bin and not have_model_xml) or (have_model_xml and not have_model_bin)):
        print("Error - only one of model_bin and model_xml were specified.  You must specify both or neither.")
        return False

    if (run_async == False) and (simultaneous_infer_per_thread != 1):
        print("Warning - If run_async is False then num_simultaneous_inferences_per_thread must be 1.")
        print("Setting num_simultaneous_inferences_per_thread to 1")
        simultaneous_infer_per_thread = 1

    return True


def print_arg_vals():

    print("")
    print("--------------------------------------------------------")
    print("Current date and time: " + str(datetime.datetime.now()))
    print("")
    print("program arguments:")
    print("------------------")
    print('num_devices: ' + str(number_of_devices))
    print('num_inferences: ' + str(number_of_inferences))
    print('num_threads_per_device: ' + str(threads_per_dev))
    print('num_simultaneous_inferences_per_thread: ' + str(simultaneous_infer_per_thread))
    print('report_interval: ' + str(report_interval))
    print('model_xml: ' + str(model_xml_fullpath))
    print('model_bin: ' + str(model_bin_fullpath))
    print('run_async: ' + str(run_async))
    #print('async_algo: ' + async_algo)
    print('time_threads: ' + str(time_threads))
    print('time_main: ' + str(time_main))
    print("--------------------------------------------------------")


def print_usage():
    print('\nusage: ')
    print('python3 classifier_performance [help][num_devices=<number of devices to use>] [num_inference=<number of inferences per device>]')
    print('')
    print('options:')
    print("  num_devices or nd - The number of devices to use for inferencing while running the to get performance ")
    print("                      The value must be between 1 and the total number of devices in the system.")
    print("                      Default is to use 1 device. ")
    print("  num_inferences or ni - The number of inferences to run on each device. ")
    print("                         Default is to run 200 inferences. ")
    print("  report_interval or ri - Report the current FPS every time this many inferences are complete. To surpress reporting set to 0")
    print("                         Default is to report FPS ever 400 inferences. ")
    print("  num_threads_per_device or ntpd - The number of threads to create that will run inferences in parallel for each device. ")
    print("                                   Default is to create 2 threads per device. ")
    print("  num_simultaneous_inferences_per_thread or nsipt - The number of inferences that each thread will create asynchronously. ")
    print("                                                    This should be at least equal to the number of NCEs on board or more.")
    print("                                                    Default is 4 simultaneous inference per thread.")
    print("  model_xml or mx - Full path to the model xml file generated by the model optimizer. ")
    print("                    Default is ./googlenet_mx_val_40000.xml ")
    print("  model_bin or mb - Full path to the model bin file generated by the model optimizer. ")
    print("                    Default is ./googlenet_mx_val_40000.bin ")
    print("  run_async or ra - Set to true to run asynchronous inferences using two threads per device")
    print("                    Default is True ")
    print("  time_main or tm - Set to true to use the time and calculate FPS from the main loop")
    print("                    Default is False ")
    print("  time_threads or tt - Set to true to use the time and calculate FPS from the time reported from inference threads")
    print("                       Default is True ")
    #print("  async_algo or aa - Set to 'complete' to use one plugin per thread or 'experiment' to use a single plugin")
    #print("                     Default is complete ")


def preprocess_image(n:int, c:int, h:int, w:int, image_filename:str) :
    image1 = cv2.imread(image_filename)
    image1 = cv2.resize(image1, (w, h))
    image1 = image1.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image1 = image1.reshape((n, c, h, w))
    return image1


def main():
    """Main function for the program.  Everything starts here.

    :return: None
    """

    if (handle_args() != True):
        print_usage()
        exit()

    print_arg_vals()

    num_ncs_devs = number_of_devices

    inferences_per_thread = int(number_of_inferences / ((threads_per_dev * num_ncs_devs)))
    inferences_per_thread = int(inferences_per_thread / simultaneous_infer_per_thread) * simultaneous_infer_per_thread
    total_number_threads = num_ncs_devs * threads_per_dev

    infer_result_queue = queue.Queue(50)

    result_times_list = [None] * (num_ncs_devs * threads_per_dev)
    thread_list = [None] * (num_ncs_devs * threads_per_dev)

    start_barrier = threading.Barrier(num_ncs_devs*threads_per_dev+1)
    end_barrier = threading.Barrier(num_ncs_devs*threads_per_dev+1)

    sync_or_async = ""


    # load a single plugin for the application
    plugin = IEPlugin(device="MYRIAD")
    net = IENetwork.from_ir(model=model_xml_fullpath, weights=model_bin_fullpath)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.inputs[input_blob].shape

    images = [None]*threads_per_dev
    images_top_results = [None]*threads_per_dev

    for thread_index in range(0, threads_per_dev, 2):
        image1 = preprocess_image(n,c,h,w,"../../data/images/nps_electric_guitar.png")
        image1_expected_top_result = 546
        image2 = preprocess_image(n,c,h,w,"../../data/images/nps_baseball.png")
        image2_expected_top_result = 429
        images[thread_index] = image1
        images_top_results[thread_index] = image1_expected_top_result
        if (thread_index + 1 < threads_per_dev):
            images[thread_index+1] = image2
            images_top_results[thread_index+1] = image2_expected_top_result

    #print("net_config: ")
    #print(net_config)

    exec_net_list = [None] * num_ncs_devs

    for dev_index in range(0, num_ncs_devs):
        # create one executable network for each device in the system
        # create 4 requests for each executable network, two for each NCE
        #exec_net_list[dev_index] = plugin.load(network=net, num_requests=threads_per_dev*simultaneous_infer_per_thread, config=net_config)
        exec_net_list[dev_index] = plugin.load(network=net, num_requests=threads_per_dev*simultaneous_infer_per_thread)

        # create threads for each executable network (one executable network per device)
        for dev_thread_index in range(0,threads_per_dev):
            total_thread_index = dev_thread_index + (threads_per_dev*dev_index)
            if (run_async):
                thread_list[total_thread_index] = threading.Thread(target=infer_async_thread_proc,
                                                                              args=[exec_net_list[dev_index], dev_thread_index*simultaneous_infer_per_thread,
                                                                                    images[dev_thread_index],
                                                                                    inferences_per_thread,
                                                                                    result_times_list, total_thread_index,
                                                                                    start_barrier, end_barrier, simultaneous_infer_per_thread,
                                                                                    images_top_results[dev_thread_index], infer_result_queue])
            else:
                thread_list[total_thread_index] = threading.Thread(target=infer_sync_thread_proc,
                                                                   args=[exec_net_list[dev_index], dev_thread_index*simultaneous_infer_per_thread,
                                                                         images[dev_thread_index],
                                                                         inferences_per_thread,
                                                                         result_times_list, total_thread_index,
                                                                         start_barrier, end_barrier,
                                                                         images_top_results[dev_thread_index], infer_result_queue])

    del net


    #start the threads
    for one_thread in thread_list:
        one_thread.start()

    start_barrier.wait()

    # save the main starting time
    main_start_time = time.time()

    print("Inferences started.")

    result_counter = 0
    for infer_result_index in range (0, total_number_threads * inferences_per_thread):
        infer_res = infer_result_queue.get(True, 10.0)
        result_counter += 1
        if (report_interval > 0):
            if ((result_counter % report_interval) == 0):
                cur_time = time.time()
                cur_duration = cur_time - main_start_time
                cur_fps = result_counter / cur_duration
                print ("after " + str(result_counter) + " inferences, FPS: " + str(cur_fps) + " FPS per device: " + str(cur_fps / num_ncs_devs))

    #print("result: " + str(infer_res))

    # wait for all the inference threads to reach end barrier
    end_barrier.wait()

    # save main end time
    main_end_time = time.time()
    print("Inferences finished.")

    for one_thread in thread_list:
        one_thread.join()

    total_thread_fps = 0.0
    total_thread_time = 0.0

    for thread_index in range(0, (num_ncs_devs*threads_per_dev)):
        #print("thread " + str(thread_index) + " time for " + str(inferences_per_thread) + " is : " + str(result_times_list[thread_index]))
        #print("thread " + str(thread_index) + " FPS is : " + str(inferences_per_thread/result_times_list[thread_index]))
        total_thread_time += result_times_list[thread_index]
        total_thread_fps += ((inferences_per_thread) / result_times_list[thread_index])


    devices_count = str(number_of_devices)


    if (time_threads):
        print("\n------------------- Thread timing -----------------------")
        print("--- Total FPS: " + str(total_thread_fps))
        print("--- FPS per device: " + str(total_thread_fps/num_ncs_devs))
        print("---------------------------------------------------------")

    main_time = main_end_time - main_start_time

    if (time_main):
        main_fps = (number_of_inferences) / (main_end_time - main_start_time)
        print ("\n------------------ Main timing -------------------------")
        print ("--- FPS: " + str(main_fps))
        print ("--- FPS per device: " + str(main_fps/num_ncs_devs))
        print ("--------------------------------------------------------")

    # clean up
    for one_exec_net in exec_net_list:
        del one_exec_net
    del plugin


# use this thread proc to try to implement:
#  1 plugin per app
#  1 executable Network per device
#  multiple threads per executable network
#  multiple requests per executable network per thread
def infer_async_thread_proc(exec_net: ExecutableNetwork, first_request_index: int,
                            image: numpy.ndarray,
                            num_total_inferences: int, result_list: list, result_index:int,
                            start_barrier: threading.Barrier, end_barrier: threading.Barrier,
                            simultaneous_infer_per_thread:int, expected_top_result:int, infer_result_queue:queue.Queue):

    #print("in new_infer_request_async_thread_proc (ASYNC)")

    input_blob = 'data'
    out_blob = 'prob'

    # sync with the main start barrier
    start_barrier.wait()

    start_time = time.time()
    end_time = start_time

    handle_list = [None]*simultaneous_infer_per_thread

    for outer_index in range(0, int(num_total_inferences/simultaneous_infer_per_thread)):

        # Start the simultaneous async inferences
        for start_index in range(0, simultaneous_infer_per_thread):
            handle_list[start_index] = exec_net.start_async(request_id=first_request_index+start_index, inputs={input_blob: image})

        # Wait for the simultaneous async inferences to finish.
        for wait_index in range(0, simultaneous_infer_per_thread):
            infer_stat = handle_list[wait_index].wait()
            res = handle_list[wait_index].outputs[out_blob]
            top_ind = numpy.argsort(res, axis=1)[0, -1:][::-1]
            infer_result_queue.put(top_ind[0], True, 10)
            # just make sure that each inference matches the expected result, if not print a warning
            #if (top_ind != expected_top_result):
            #    print("Warning - thread " + str(result_index) + ", got wrong result - top_ind was " + str(top_ind[0]) + " expected: " + str(expected_top_result) + " infer_stat: " + str(infer_stat))

            handle_list[wait_index] = None


    # save the time spent on inferences within this inference thread and associated reader thread
    end_time = time.time()
    total_inference_time = end_time - start_time
    result_list[result_index] = total_inference_time

    # wait for all inference threads to finish
    end_barrier.wait()

# use this thread proc to try to implement:
#  1 plugin per app
#  1 executable Network per device
#  multiple threads per executable network
#  multiple requests per executable network per thread
def infer_sync_thread_proc(exec_net: ExecutableNetwork, first_request_index: int,
                           image: numpy.ndarray,
                           num_total_inferences: int, result_list: list, result_index:int,
                           start_barrier: threading.Barrier, end_barrier: threading.Barrier,
                           expected_top_result:int, infer_result_queue:queue.Queue):

    #print("in new_infer_request_sync_thread_proc (SYNC)")

    input_blob = 'data'
    out_blob = 'prob'

    # sync with the main start barrier
    start_barrier.wait()

    start_time = time.time()
    end_time = start_time

    exec_net_reqs = exec_net.requests
    req_to_use_1 = exec_net_reqs[first_request_index]

    for index in range(0, num_total_inferences):
        #res = exec_net.infer(inputs={input_blob: image})
        req_to_use_1.infer(inputs={input_blob: image})
        res = req_to_use_1.outputs[out_blob]
        top_ind = numpy.argsort(res, axis=1)[0, -1:][::-1]
        infer_result_queue.put(top_ind[0], True, 10)

        # just make sure that each inference matches the expected result, if not print a warning
        #if (top_ind != expected_top_result):
        #    print("Warning - thread " + str(result_index) + ", got wrong result - top_ind was " + str(top_ind[0]) + " expected: " + str(expected_top_result) )



    # save the time spent on inferences within this inference thread and associated reader thread
    end_time = time.time()
    total_inference_time = end_time - start_time
    result_list[result_index] = total_inference_time

    # wait for all inference threads to finish
    end_barrier.wait()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
