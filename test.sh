#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_position_wise_feed_forward tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=keras_position_wise_feed_forward --with-doctest
