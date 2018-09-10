#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# pulls images from file system and places them in a Queue or starts an inference for them on a network processor

import cv2
import queue
import threading
import time
from googlenet_processor import GoogleNetProcessor
from queue import Queue
import os
import numpy


class ImageProcessor:
    """Class that pulls images from the files system and either starts an inference with them or
    puts them on a queue depending on how the instance is constructed.
    """

    def __init__(self, image_dir_path:str, request_image_width:int=640, request_image_height:int = 480,
                 request_image_mean = [0.0, 0.0, 0.0],
                 network_processor_list:list=None, output_queue:Queue=None, queue_put_wait_max:float = 0.01,
                 queue_full_sleep_seconds:float = 0.1):
        """Initializer for the class.

        :param image_dir_path: directory path for the directory in which image files will be used.
        :param request_image_width: the width in pixels to resize the images to
        :param request_image_height: the height in pixels to resize the images to.
        :param network_processor_list: list of neural network processors on which we will start inferences for each frame.
        If a value is passed for this parameter then the output_queue, queue_put_wait_max, and
        queue_full_sleep_seconds will be ignored and should be None
        :param output_queue: A queue on which the images will be placed if the network_processor is None
        :param queue_put_wait_max: The max number of seconds to wait when putting on output queue
        :param queue_full_sleep_seconds: The number of seconds to sleep when the output queue is full.
        """
        self._queue_full_sleep_seconds = queue_full_sleep_seconds
        self._queue_put_wait_max = queue_put_wait_max
        self._image_dir_path = image_dir_path
        self._request_image_width = request_image_width
        self._request_image_height = request_image_height
        self._pause_mode = False

        # save the image preprocessing params
        self._request_image_width = request_image_width
        self._request_image_height = request_image_height
        self._request_image_mean = request_image_mean

        self._output_queue = output_queue
        self._network_processor_list = network_processor_list

        self._use_output_queue = False
        if (not(self._output_queue is None)):
            self._use_output_queue = True

        self._worker_thread = None

        # get list of all the .jpg files in the image directory
        input_image_filename_list = os.listdir(self._image_dir_path)
        input_image_filename_list = [self._image_dir_path + '/' + a_file for a_file in input_image_filename_list if
                                     a_file.endswith('.jpg')]

        if (len(input_image_filename_list) < 1):
            # no images to show
            print('No .jpg files found')
            raise Exception("No image files found in " + self._image_dir_path)
            return 1

        self._preprocessed_image_list = list()
        self._original_image_list = list()

        for input_image_file in input_image_filename_list:
            # Read image from file, resize it to network width and height
            # save a copy in img_cv for display, then convert to float32, normalize (divide by 255),
            # and finally convert to convert to float16 to pass to LoadTensor as input for an inference
            input_image = cv2.imread(input_image_file)


            standardized_size_original_image = cv2.resize(input_image,
                                         (640,
                                          480),
                                          cv2.INTER_LINEAR)
            self._original_image_list.append(standardized_size_original_image)


            # resize image to network width and height
            # then convert to float32, normalize (divide by 255),
            # and finally convert to float16 to pass to LoadTensor as input
            # for an inference
            # this returns a new image so the input_image is unchanged
            inference_image = cv2.resize(input_image,
                                         (self._request_image_width,
                                          self._request_image_height),
                                          cv2.INTER_LINEAR)
            inference_image = inference_image.astype(numpy.float32)
            inference_image[:, :, 0] = (inference_image[:, :, 0] - self._request_image_mean[0])
            inference_image[:, :, 1] = (inference_image[:, :, 1] - self._request_image_mean[1])
            inference_image[:, :, 2] = (inference_image[:, :, 2] - self._request_image_mean[2])

            self._preprocessed_image_list.append(inference_image)

    def get_request_image_width(self):
        """ get the width of the images that will be placed on queue or sent to neural network processor.
        :return: the width the images will be resized to
        """
        return self._request_image_width

    # the
    def get_request_image_height(self):
        """get the height of the images that will be put in the queue or sent to the neural network processor

        :return: The height the images will be resized to
        """
        return self._request_image_height


    def start_processing(self):
        """Starts the asynchronous thread reading image list and placing images in the output queue or sending to the
        neural network processor

        :return: None
        """
        self._end_flag = False
        if (self._use_output_queue):
            if (self._worker_thread == None):
                self._worker_thread = threading.Thread(target=self._do_work_queue, args=())
        else:
            if (self._worker_thread == None):
                self._worker_thread = threading.Thread(target=self._do_work_network_processor, args=())

        self._worker_thread.start()


    def stop_processing(self):
        """stops the asynchronous thread from reading any new frames from image list

        :return:
        """
        if (self._end_flag == True):
            # Already stopped
            return

        self._end_flag = True

    def is_processing(self):
        """ Determine if still processing or if processing has ended

        :return: True is returned if still processing or False if not
        """
        return not(self._end_flag)

    def pause(self):
        """pauses the aysnchronous processing so that it will not read any new frames until unpause is called.
        :return: None
        """
        self._pause_mode = True


    def unpause(self):
        """ Unpauses the asynchronous processing that was previously paused by calling pause

        :return: None
        """
        self._pause_mode = False


    def _do_work_queue(self):
        """Thread target.  When call start_processing and initialized with an output queue,
           this function will be called in its own thread.  it will keep working until stop_processing is called.
           or an error is encountered.  If the neural network processor was passed to the initializer rather than
           a queue then this function will not be called.

        :return: None
        """
        print('in image_processor worker thread')
        if ((self._preprocessed_image_list == None) or (len(self._preprocessed_image_list) < 1)):
            print('image_processor no images in the preprocessed list, returning.')
            return

        image_index = 0

        while (not self._end_flag):
            try:
                while (self._pause_mode):
                    time.sleep(0.1)

                preprocessed_image = self._preprocessed_image_list[image_index]
                image_index += 1
                self._output_queue.put(preprocessed_image, True, self._queue_put_wait_max)

            except queue.Full:
                # if our output queue is full sleep a little while before
                # trying the next image from the list.
                time.sleep(self._queue_full_sleep_seconds)

        print('exiting image_processor worker thread for queue')


    def _do_work_network_processor(self):
        """Thread target.  when call start_processing and initialized with an neural network processor,
           this function will be called in its own thread.  it will keep working until stop_processing is called.
           or an error is encountered.  If the initializer was called with a queue rather than a neural network
           processor then this will not be called.

        :return: None
        """
        print('in image_processor worker thread for network.')
        if ((self._preprocessed_image_list == None) or (len(self._preprocessed_image_list) < 1)):
            print('No preprocessed images, returning.')
            return

        image_index = 0

        while (not self._end_flag):
            done = False
            try:
                while (self._pause_mode):
                    time.sleep(0.1)

                # give each nework processor a frame to process
                for one_network_proc in self._network_processor_list:

                    preprocessed_image = self._preprocessed_image_list[image_index]

                    # wait until there is room on the network processor's input queue.
                    # if we don't do this and clean up is called from another thread then
                    # there is a race condition
                    while (one_network_proc.is_input_queue_full() and (not (self._end_flag))):
                        time.sleep(0.1)

                    if (not (self._end_flag)):
                        one_network_proc.start_preprocessed_aysnc_inference(preprocessed_image, self._original_image_list[image_index])
                        image_index += 1
                        if (image_index >= len(self._preprocessed_image_list)):
                            image_index = 0
                    else:
                        done=True
                        break

                if (done) : break

            except Exception:
                # If our output queue is full sleep a little while before
                # trying the next image.
                print("Exception occurred writing to the neural network processor.")
                raise

        self._end_flag = True
        print('exiting image_processor worker thread for network processor')


    def cleanup(self):
        """Should be called once for each class instance when the class consumer is finished with it.

        :return: None
        """
        # wait for worker thread to finish if it still exists
        if (not(self._worker_thread is None)):
            self._worker_thread.join()
            self._worker_thread = None

