#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.
# NPS

# Object detector using SSD Mobile Net

from mvnc import mvncapi as mvnc
import numpy as numpy
import cv2
import time
import threading


class GoogleNetProcessor:

    # Neural network assumes input images are these dimensions.
    NN_IMAGE_WIDTH = 224
    NN_IMAGE_HEIGHT = 224

    def __init__(self, network_graph_filename: str, ncs_device: mvnc.Device,
                 name = None):
        """Initializes an instance of the class

        :param network_graph_filename: is the path and filename to the graph
               file that was created by the ncsdk compiler
        :param ncs_device: is an open ncs device object to use for inferences for this graph file
        :param name: A name to use for the processor.  Nice to have when debugging multiple instances
        on multiple threads
        :return : None
        """
        self._device = ncs_device
        self._network_graph_filename = network_graph_filename
        # Load graph from disk and allocate graph.
        try:
            with open(self._network_graph_filename, mode='rb') as graph_file:
                graph_in_memory = graph_file.read()
            self._graph = mvnc.Graph("GoogleNet Graph")
            self._fifo_in, self._fifo_out = self._graph.allocate_with_fifos(self._device, graph_in_memory)

            self._input_fifo_capacity = self._fifo_in.get_option(mvnc.FifoOption.RO_CAPACITY)
            self._output_fifo_capacity = self._fifo_out.get_option(mvnc.FifoOption.RO_CAPACITY)

        except:
            print('\n\n')
            print('Error - could not load neural network graph file: ' + network_graph_filename)
            print('\n\n')
            raise

        self._classification_labels = GoogleNetProcessor.get_classification_labels()
        self._mean = GoogleNetProcessor.get_mean()

        self._end_flag = True
        self._name = name
        if (self._name is None):
            self._name = "no name"

        # lock to let us count calls to asynchronus inferences and results
        self._async_count_lock = threading.Lock()
        self._async_inference_count = 0

    def cleanup(self, destroy_device=False):
        """Call once when done with the instance of the class

        :param destroy_device: pass True to close and destroy the neural compute device or
        False to leave it open
        :return: None
        """

        self._drain_queues()
        self._fifo_in.destroy()
        self._fifo_out.destroy()
        self._graph.destroy()

        if (destroy_device):
            self._device.close()
            self._device.destroy()


    def get_device(self):
        '''Get the device this processor is using.

        :return:
        '''
        return self._device

    def get_name(self):
        '''Get the name of this processor.

        :return:
        '''
        return self._name

    def drain_queues(self):
        """ Drain the input and output FIFOs for the processor.  This should only be called
        when its known that no calls to start_async_inference will be made during this method's
        exectuion.

        :return: None
        """
        self._drain_queues()


    def get_nn_image_width(self):
        return self.NN_IMAGE_WIDTH

    def get_nn_image_height(self):
        return self.NN_IMAGE_HEIGHT

    @staticmethod
    def get_classification_labels():
        """get a list of the classifications that are supported by this neural network

        :return: the list of the classification strings
        """
        EXAMPLES_BASE_DIR = '../../'
        gn_labels_file = EXAMPLES_BASE_DIR + 'data/ilsvrc12/synset_words.txt'
        gn_labels = numpy.loadtxt(gn_labels_file, str, delimiter='\t')
        for label_index in range(0, len(gn_labels)):
            temp = gn_labels[label_index].split(',')[0].split(' ', 1)[1]
            gn_labels[label_index] = temp
        return gn_labels

    @staticmethod
    def get_mean():
        """get the network mean

        :return:
        """
        EXAMPLES_BASE_DIR = '../../'
        gn_mean = numpy.load(EXAMPLES_BASE_DIR + 'data/ilsvrc12/ilsvrc_2012_mean.npy').mean(1).mean(1)
        return gn_mean

    def start_aysnc_inference(self, input_image:numpy.ndarray):
        """Start an asynchronous inference.  When its complete it will go to the output FIFO queue which
           can be read using the get_async_inference_result() method
           If there is no room on the input queue this function will block indefinitely until there is room,
           when there is room, it will queue the inference and return immediately

        :param input_image: he image on which to run the inference.
             it can be any size but is assumed to be opencv standard format of BGRBGRBGR...
        :return: None
        """

        # resize image to network width and height
        # then convert to float32, normalize (divide by 255),
        # and finally convert to float16 to pass to LoadTensor as input
        # for an inference
        # this returns a new image so the input_image is unchanged
        inference_image = cv2.resize(input_image,
                                 (GoogleNetProcessor.NN_IMAGE_WIDTH,
                                  GoogleNetProcessor.NN_IMAGE_HEIGHT),
                                 cv2.INTER_LINEAR)

        inference_image = inference_image.astype(numpy.float32)
        inference_image[:,:,0] = (inference_image[:,:,0] - self._mean[0])
        inference_image[:,:,1] = (inference_image[:,:,1] - self._mean[1])
        inference_image[:,:,2] = (inference_image[:,:,2] - self._mean[2])

        self._inc_async_count()

        # Load tensor and get result.  This executes the inference on the NCS
        self._graph.queue_inference_with_fifo_elem(self._fifo_in, self._fifo_out, inference_image, input_image)

        return


    def start_preprocessed_aysnc_inference(self, preprocessed_inference_image:numpy.ndarray, original_image:numpy.ndarray):
        """Start an asynchronous inference.  When its complete it will go to the output FIFO queue which
           can be read using the get_async_inference_result() method
           If there is no room on the input queue this function will block indefinitely until there is room,
           when there is room, it will queue the inference and return immediately

        :param input_image: he image on which to run the inference.
             it can be any size but is assumed to be opencv standard format of BGRBGRBGR...
        :return: None
        """

        self._inc_async_count()

        # Load tensor and get result.  This executes the inference on the NCS
        # when read from the fifo the original image will be the user object.
        self._graph.queue_inference_with_fifo_elem(self._fifo_in, self._fifo_out, preprocessed_inference_image, original_image)

        return


    def _inc_async_count(self):
        self._async_count_lock.acquire()
        self._async_inference_count += 1
        self._async_count_lock.release()

    def _dec_async_count(self):
        self._async_count_lock.acquire()
        self._async_inference_count -= 1
        self._async_count_lock.release()

    def _get_async_count(self):
        self._async_count_lock.acquire()
        ret_val = self._async_inference_count
        self._async_count_lock.release()
        return ret_val


    def get_async_inference_result(self):
        """Reads the next available object from the output FIFO queue.  If there is nothing on the output FIFO,
        this fuction will block indefinitiley until there is.

        :return: tuple of the following items:
            index of the top probability classification
            label that corresponds to the index of the top probablity classification
            probability of the top classification
            the orginal image that was passed as part of start_async_inference.
        """

        self._dec_async_count()

        # get the result from the queue
        output, input_image = self._fifo_out.read_elem()

        # save original width and height
        #input_image_width = input_image.shape[1]
        #input_image_height = input_image.shape[0]
        order = output.argsort()[::-1][:1]

        '''
        print('\n------- prediction --------')
        for i in range(0, 1):
            print('prediction ' + str(i) + ' (probability ' + str(output[order[i]]) + ') is ' + labels[
                order[i]] + '  label index is: ' + str(order[i]))
        '''

        # index, label, probability
        ret_index = order[0]
        ret_label = self._classification_labels[order[0]]
        ret_prob = output[order[0]]

        return ret_index, ret_label, ret_prob, input_image


    def is_input_queue_empty(self):
        """Determines if the input queue for this instance is empty

        :return: True if input queue is empty or False if not.
        """
        count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        return (count == 0)


    def is_input_queue_full(self):
        """Determines if the input queue is full

        :return: True if the input queue is full and calls to start_async_inference would block
        or False if the queue is not full and start_async_inference would not block
        """
        count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        return ((self._input_fifo_capacity - count) == 0)


    def _drain_queues(self):
        """ Drain the input and output FIFOs for the processor.  This should only be called
        when its known that no calls to start_async_inference will be made during this method's
        exectuion.

        :return: None.
        """
        in_count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
        count = 0

        while (self._get_async_count() != 0):
            count += 1
            if (out_count > 0):
                self.get_async_inference_result()
                out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
            else:
                time.sleep(0.1)

            in_count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
            out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
            if (count > 3):
                blank_image = numpy.zeros((self.NN_IMAGE_HEIGHT, self.NN_IMAGE_WIDTH, 3),
                                          numpy.float32)
                self.do_sync_inference(blank_image)

            if (count == 30):
                # should really not be nearly this high of a number but working around an issue in the
                # ncapi where an inferece can get stuck in process
                raise Exception("Could not drain FIFO queues for '" + self._name + "'")

        in_count = self._fifo_in.get_option(mvnc.FifoOption.RO_WRITE_FILL_LEVEL)
        out_count = self._fifo_out.get_option(mvnc.FifoOption.RO_READ_FILL_LEVEL)
        return


    def do_sync_inference(self, input_image:numpy.ndarray):
        """Do a single inference synchronously.
        Don't mix this with calls to get_async_inference_result, Use one or the other.  It is assumed
        that the input queue is empty when this is called which will be the case if this isn't mixed
        with calls to get_async_inference_result.

        :param input_image: the image on which to run the inference it can be any size.
        :return: a tuple of the following:
            index for top 1 result,
            label corresonding to the index for the top 1 result,
            probability of the top 1 result.
        """
        self.start_aysnc_inference(input_image)
        index, label, prob, original_image = self.get_async_inference_result()

        return index, label, prob






