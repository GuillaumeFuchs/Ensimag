all : 
	cd src/ && $(MAKE) all

docs : 
	doxygen doc/doxygen.cfg
	#cd doc/ && $(MAKE) docs

clean : 
	cd src/ && $(MAKE) clean

realclean :
	cd src/ && $(MAKE) clean && rm couv pricer pricert test-parser delta
	cd doc/ && rm -rf html/

check :
	cd src/ && ./test_parser.sh && ./test_pricer.sh && ./test_delta.sh
