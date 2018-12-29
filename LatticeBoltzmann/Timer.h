///////////////////////////////////////////////////////////////////////////////////////////////////
/**
* \file       Timer.h
* \author     Martin Cap
* \date       2018/12/23
* \brief      Defines the Timer class.
*
*  Defines the Timer class which is used for basic timing and logging.
*
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

#include "Config.h"

using namespace std;

typedef chrono::time_point<chrono::high_resolution_clock> Timepoint;

class Timer {
public:

	Timepoint startTime;
	Timepoint lastClockTime;

	string configString;

	double accumulatedTime;
	double avgTime;
	int numMeasurementsForAvg;
	int measurementCount;

	bool running;

	bool logToFile;
	bool printToConsole;

	ofstream logFile;

	Timer(bool logToFile = false, bool printToConsole = true, int numMeasurementsForAvg = 1000);
	~Timer();

	void start();
	void clockAvgStart();
	bool clockAvgEnd();
	void end(bool printResults = false);


};

