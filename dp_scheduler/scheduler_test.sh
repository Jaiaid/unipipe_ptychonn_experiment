#!/bin/bash

gcc scheduler.c -o scheduler_test

time ./scheduler_test < test1.txt > test1.out
time ./scheduler_test < test2.txt > test2.out
time ./scheduler_test < test3.txt > test3.out
time ./scheduler_test < test4.txt > test4.out
time ./scheduler_test < test5.txt > test5.out