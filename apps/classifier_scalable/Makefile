
ifneq ($(findstring movidius, $(PYTHONPATH)), movidius)
	export PYTHONPATH:=/opt/movidius/caffe/python:$(PYTHONPATH)
endif

NCCOMPILE = mvNCCompile
NCPROFILE = mvNCProfile
NCCHECK   = mvNCCheck

# filenames for the graph files that we'll copy to this directory.
CLASSIFIER_GRAPH_FILENAME = graph

.PHONY: all
all: data deps

.PHONY: googlenet
googlenet: 
	@echo "\nmaking googlenet"
	(cd ../../caffe/GoogLeNet; make compile; cd ../../apps/classifier_scalable; cp ../../caffe/GoogLeNet/graph ./${CLASSIFIER_GRAPH_FILENAME};)


.PHONY: data
data: 
	@echo "\nmaking data"
	(cd ../../data/ilsvrc12; make all; cd ../../apps/classifier_scalable;)

.PHONY: deps
deps: googlenet
	@echo "/nmaking deps"


.PHONY: run_py
run_py: data deps
	@echo "\nmaking run_py"
	python3 ./classifier_scalable.py


.PHONY: run
run: run_py

.PHONY: install-reqs
install-reqs: 
	@echo "\nmaking install-reqs"
	./install-opencv-from_source.sh

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes everything needed to run but doesn't run";
	@echo "  make googlenet - compiles and copies the graph file";
	@echo "  make run - runs the application";
	@echo "  make install-reqs - Installs requirements on your system."
	@echo "                      Will remove pip3 opencv and build from source and install a new version." ;
	@echo "                      Only needed once on your system and only if the its not already installed.";
	@echo "                      This may take a long time depending on your system.";
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${CLASSIFIER_GRAPH_FILENAME}



