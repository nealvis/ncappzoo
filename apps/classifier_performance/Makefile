
# filenames for the graph files that we'll copy to this directory.
CLASSIFIER_GRAPH_BASE_FILENAME = bvlc_googlenet
CLASSIFIER_RELATIVE_DIR = ../../caffe/GoogLeNet

.PHONY: all
all: data deps

.PHONY: classifier_model
classifier_model: 
	@echo "\nmaking classifier"
	@if [ -e ${CLASSIFIER_GRAPH_BASE_FILENAME}.xml ] ; \
	then \
		echo "${CLASSIFIER_GRAPH_BASE_FILENAME}.xml file already exists, skipping compiliation"; \
	else \
		(cd ${CLASSIFIER_RELATIVE_DIR}; make compile_model; cd ../../apps/classifier_performance; cp ${CLASSIFIER_RELATIVE_DIR}/${CLASSIFIER_GRAPH_BASE_FILENAME}.* . ; ); \
	fi  

	


.PHONY: data
data: 
	@echo "\nmaking data"
	(cd ../../data/ilsvrc12; make all; cd ../../apps/classifier_performance;)

.PHONY: deps
deps: classifier_model
	@echo "/nmaking deps"


.PHONY: run_py
run_py: data deps
	@echo "\nmaking run_py"
	python3 ./classifier_performance.py


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
	@echo "  make classifier_model - compiles a classifier model to use when running";
	@echo "  make run - runs the application";
	@echo "  make install-reqs - Installs requirements on your system."
	@echo "                      Will remove pip3 opencv and build from source and install a new version." ;
	@echo "                      Only needed once on your system and only if the its not already installed.";
	@echo "                      This may take a long time depending on your system.";
	@echo "  make clean - removes all created content"

.PHONY: clean
clean: 
	@echo "\nmaking clean"
	rm -f ${CLASSIFIER_GRAPH_BASE_FILENAME}.*



