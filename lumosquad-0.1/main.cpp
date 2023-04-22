///////////////////////////////////////////////////////////////////////////////////
// File : main.cpp
///////////////////////////////////////////////////////////////////////////////////
//
// LumosQuad - A Lightning Generator
// Copyright 2007
// The University of North Carolina at Chapel Hill
//
///////////////////////////////////////////////////////////////////////////////////
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//  The University of North Carolina at Chapel Hill makes no representations
//  about the suitability of this software for any purpose. It is provided
//  "as is" without express or implied warranty.
//
//  Permission to use, copy, modify and distribute this software and its
//  documentation for educational, research and non-profit purposes, without
//  fee, and without a written agreement is hereby granted, provided that the
//  above copyright notice and the following three paragraphs appear in all
//  copies.
//
//  THE UNIVERSITY OF NORTH CAROLINA SPECIFICALLY DISCLAIM ANY WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
//  FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN
//  "AS IS" BASIS, AND THE UNIVERSITY OF NORTH CAROLINA HAS NO OBLIGATION TO
//  PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
//
//  Please send questions and comments about LumosQuad to kim@cs.unc.edu.
//
///////////////////////////////////////////////////////////////////////////////////
//
//  This program uses OpenEXR, which has the following restrictions:
//
//  Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
//  Digital Ltd. LLC
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are
//  met:
//  *       Redistributions of source code must retain the above copyright
//  notice, this list of conditions and the following disclaimer.
//  *       Redistributions in binary form must reproduce the above
//  copyright notice, this list of conditions and the following disclaimer
//  in the documentation and/or other materials provided with the
//  distribution.
//  *       Neither the name of Industrial Light & Magic nor the names of
//  its contributors may be used to endorse or promote products derived
//  from this software without specific prior written permission.
//
///////////////////////////////////////////////////////////////////////////////////
///
/// \mainpage Fast Animation of Lightning Using An Adaptive Mesh
/// \section Introduction
///
/// This project is an implementation of the paper
/// <b><em>Fast Animation of Lightning Using An Adaptive Mesh</em></b>. It
/// includes both the simulation and rendering components described in that
/// paper.
///
/// Several pieces of software are used in this project that the respective
/// authors were kind enough to make freely available:
///
/// <UL>
/// <LI> <A HREF =
/// "http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html">Mersenne
/// twister</A>
///      (Thanks to Makoto Matsumoto)
/// <LI> <A HREF = "http://www.fftw.org/">FFTW</A>
///      (Thanks to Matteo Frigo and Steven G. Johnson)
/// <LI> <A HREF = "http://www.openexr.com/">OpenEXR</A>
///      (Thanks to ILM)
/// <LI> <A HREF = "http://www.cs.unc.edu/~walk/software/glvu/">GLVU</A>
///      (Thanks to Walkthru)
/// <LI> <A HREF = "http://www.cs.virginia.edu/~gfx/pubs/antimony/">Antimony</A>
///      (Thanks to Daniel Dunbar and Greg Humphreys)
/// </UL>
///
/// <em>Theodore Kim, kim@cs.unc.edu, October 2006</em>
///
///////////////////////////////////////////////////////////////////////////////////

#define COMMAND_LINE_VERSION 1

#include <iostream>

#include "APSF.h"
#include "EXR.h"
#include "FFT.h"
#include "QUAD_DBM_2D.h"
#include "ppm/ppm.hpp"

using namespace std;

////////////////////////////////////////////////////////////////////////////
// globals
////////////////////////////////////////////////////////////////////////////
int iterations = 10;
static QUAD_DBM_2D* potential = new QUAD_DBM_2D(256, 256, iterations);
APSF apsf(512);

bool include_noise = false;

// input image info
int inputWidth = -1;
int inputHeight = -1;

// input params
string inputFile;
string outputFile;

// image scale
float scale = 5;

// pause the simulation?
bool pause = false;
#include <fstream>

int do_convolve(std::string in, std::string out, int N) {
    float* buf = new float[N * N];
    std::ifstream infile(in);
    infile.read(reinterpret_cast<char*>(buf), N * N * sizeof(float));
    infile.close();

    APSF apsf(N * 2);
    apsf.generateKernelFast();
    bool success = FFT::convolve(buf, apsf.kernel(), N, N,
                                 apsf.res(), apsf.res());

    if (!success) {
        std::cerr << "Error: convolution failed." << std::endl;
        return 1;
    }

    std::ofstream outfile(out);
    if (!outfile) {
        std::cerr << "Error: " << out << " is not a valid file." << std::endl;
        return 1;
    }

    outfile.write(reinterpret_cast<char*>(buf), N * N * sizeof(float));
    outfile.close();
    delete[] buf;

    return 0;
}

////////////////////////////////////////////////////////////////////////////
// render the glow
////////////////////////////////////////////////////////////////////////////
void renderGlow(string filename, int scale = 1) {
    int w = potential->xDagRes() * scale;
    int h = potential->yDagRes() * scale;

    std::cout << w << std::endl;

    // draw the DAG
    float*& source = potential->renderOffscreen(scale);

    // if there is no input dimensions specified, else there were input
    // image dimensions, so crop it
    if (inputWidth == -1) {
        inputWidth = potential->inputWidth();
        inputHeight = potential->inputHeight();
    }

    // copy out the cropped version
    int wCropped = inputWidth * scale;
    int hCropped = inputHeight * scale;
    float* cropped = new float[wCropped * hCropped];
    cout << endl
         << " Generating EXR image width: " << wCropped
         << " height: " << hCropped << endl;
    for (int y = 0; y < hCropped; y++)
        for (int x = 0; x < wCropped; x++) {
            int uncroppedIndex = x + y * w;
            int croppedIndex = x + y * wCropped;
            cropped[croppedIndex] = source[uncroppedIndex];
        }

    // create the filter
    apsf.generateKernelFast();

    // convolve with FFT
    bool success = FFT::convolve(cropped, apsf.kernel(), wCropped, hCropped,
                                 apsf.res(), apsf.res());

    // save cropped to file
    std::ofstream croppedFile2(filename + "_result.bin");
    croppedFile2.write((char*)cropped, sizeof(float) * wCropped * hCropped);
    croppedFile2.close();

    // // save aspf.kernel to file
    // auto kernel = apsf.kernel();
    // std::ofstream kernelFile("kernel.bin");
    // kernelFile.write((char*)kernel, sizeof(float) * apsf.res() * apsf.res());
    // std::cout << "yup" << std::endl;
    // std::cout << apsf.res() << std::endl;

    // do_convolve("../big/out.bin", "foo.bin");

    // if (success) {
    // EXR::writeEXR(filename.c_str(), cropped, wCropped, hCropped);
    // cout << " " << filename << " written." << endl;
    // } else
    //     cout << " Final image generation failed." << endl;

    delete[] cropped;
}

void render_bolt(string filename) {
    int w = potential->xDagRes();
    int h = potential->yDagRes();

    unsigned char* source = new unsigned char[w * h];

    for (CELL* candidate : potential->_candidates) {
        int x = candidate->center[0] * w;
        int y = candidate->center[1] * h;
        int index = x + y * w;

        if (candidate->state == NEGATIVE) {
            source[index] = 1;
        } else if (candidate->state == EMPTY) {
            source[index] = 2;
        }
    }

    int num_changed = 0;
    bool did_change = true;
    do {
        did_change = false;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int index = x + y * w;
                if (source[index] == 0) {
                    int count = 0;
                    if (x > 0 && source[index - 1] == 1) count++;
                    if (x < w - 1 && source[index + 1] == 1) count++;
                    if (y > 0 && source[index - w] == 1) count++;
                    if (y < h - 1 && source[index + w] == 1) count++;

                    if (count >= 3) {
                        source[index] = 1;
                        num_changed++;
                        did_change = true;
                    }
                }
            }
        }
    } while (did_change);

    std::ofstream outfile(filename);
    outfile.write((char*)source, w * h * sizeof(unsigned char));
    outfile.close();

    delete[] source;
}

void render_candidates(string filename) {
    int w = potential->xDagRes();
    int h = potential->yDagRes();

    float* source = new float[w * h];

    for (int i = 0; i < w * h; i++)
        source[i] = 0.0;

    for (CELL* candidate : potential->_candidates) {
        int x = candidate->center[0] * w;
        int y = candidate->center[1] * h;
        int index = x + y * w;

        if (candidate->state == EMPTY) {
            source[index] = candidate->potential;
        }
    }

    std::ofstream outfile(filename);
    outfile.write((char*)source, w * h * sizeof(float));
    outfile.close();

    delete[] source;
}

////////////////////////////////////////////////////////////////////////////
// load image file into the DBM simulation
////////////////////////////////////////////////////////////////////////////
bool loadImages(string inputFile) {
    // load the files
    unsigned char* input = NULL;
    LoadPPM(inputFile.c_str(), input, inputWidth, inputHeight);

    unsigned char* start = new unsigned char[inputWidth * inputHeight];
    unsigned char* repulsor = new unsigned char[inputWidth * inputHeight];
    unsigned char* attractor = new unsigned char[inputWidth * inputHeight];
    unsigned char* terminators = new unsigned char[inputWidth * inputHeight];

    // composite RGB channels into one
    for (int x = 0; x < inputWidth * inputHeight; x++) {
        start[x] = (input[3 * x] == 255) ? 255 : 0;
        repulsor[x] = (input[3 * x + 1] == 255) ? 255 : 0;
        attractor[x] = (input[3 * x + 2] == 255) ? 255 : 0;
        terminators[x] = 0;

        if (input[3 * x] + input[3 * x + 1] + input[3 * x + 2] == 255 * 3) {
            terminators[x] = 255;
            start[x] = repulsor[x] = attractor[x] = 0;
        }
    }

    if (potential) delete potential;
    potential = new QUAD_DBM_2D(inputWidth, inputHeight, iterations);
    bool success = potential->readImage(start, attractor, repulsor, terminators,
                                        inputWidth, inputHeight);

    // delete the memory
    delete[] input;
    delete[] start;
    delete[] repulsor;
    delete[] attractor;
    delete[] terminators;

    return success;
}

int width = 600;
int height = 600;
bool animate = false;
float camera[2];
float translate[2];

////////////////////////////////////////////////////////////////////////////
// window Reshape function
////////////////////////////////////////////////////////////////////////////
void Reshape(int w, int h) {
    if (h == 0) h = 1;

    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-camera[0] - translate[0], camera[0] + translate[0],
               -camera[1] - translate[1], camera[1] + translate[1]);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

////////////////////////////////////////////////////////////////////////////
// GLUT Display callback
////////////////////////////////////////////////////////////////////////////
void Display() {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-camera[0] + translate[0], camera[0] + translate[0],
               -camera[1] + translate[1], camera[1] + translate[1]);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    potential->draw();
    potential->drawSegments();

    glutSwapBuffers();
}

////////////////////////////////////////////////////////////////////////////
// GLUT Keyboard callback
////////////////////////////////////////////////////////////////////////////
void Keyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 'p':
            pause = !pause;
            break;

        case 'q':
            cout << " You terminated the simulation prematurely." << endl;
            exit(0);
            break;
    }

    glutPostRedisplay();
}

// #include <chrono>
// using namespace std::chrono;

// // start time
// high_resolution_clock::time_point t1;
// std::vector<double> diff;
// int i;
// void renderGlow(string filename, int scale = 1);

//   i += 1;

//   if (i % 1000 == 0) {
//     renderGlow("glow" + std::to_string(i) + ".png", 1);
//   }

//   high_resolution_clock::time_point t2 = high_resolution_clock::now();
//   auto time_span = duration_cast<nanoseconds>(t2 - t1);
//   t1 = t2;

//   diff.push_back(time_span.count());

//   double average = 0.0;
//   for (int x = 0; x < diff.size(); x++) {
//     if (x == 0) continue;
//     average += diff[x];
//   }
//   average /= (double)diff.size();
//   // cout << "Average time: " << average << endl;

int i = 0;
////////////////////////////////////////////////////////////////////////////
// window Reshape function
////////////////////////////////////////////////////////////////////////////
void Idle() {
    if (!pause)
        for (int x = 0; x < 100; x++) {
            bool success = potential->addParticle();
            if (rand() % 100 == 0) {
                // renderGlow(outputFile + std::to_string(i++), scale);
                char number[256];
                sprintf(number, "%05d", i++);
                render_candidates(outputFile + "_" + number + "_candidates");
                render_bolt(outputFile + "_" + number + "_bolt");
            }

            if (!success) {
                cout << " No nodes left to add! Is your terminator reachable?"
                     << endl;
                // exit(1);
                return;
            }

            if (potential->hitGround()) {
                glutPostRedisplay();
                cout << endl
                     << endl;

                // write out the DAG file
                string lightningFile =
                    inputFile.substr(0, inputFile.size() - 3) +
                    string("lightning");
                cout << " Intermediate file " << lightningFile << " written."
                     << endl;
                potential->writeDAG(lightningFile.c_str());

                // render the final EXR file
                renderGlow(outputFile, scale);
                delete potential;
                exit(0);
            }
        }

    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////
// GLUT Main
////////////////////////////////////////////////////////////////////////////
int glutMain() {
    float smaller = 1.0f;
    camera[0] = smaller * 0.5f;
    camera[1] = smaller * 0.5f;
    translate[0] = 0.0f;
    translate[1] = 0.0f;

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA);
    glutInitWindowPosition(50, 50);
    glutInitWindowSize(width, height);
    glutCreateWindow("Lumos: A Lightning Generator v0.1");
    glutHideWindow();

    glutDisplayFunc(Display);
    glutKeyboardFunc(Keyboard);
    glutIdleFunc(Idle);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    Reshape(width, height);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glShadeModel(GL_SMOOTH);

    // Go!
    glutMainLoop();

    return 0;
}

#if 1
////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    srand(time(NULL));
    if (argc < 5) {
        cout << endl;
        cout << "   LumosQuad <input file> <output file> <scale> <1 or 0 to enable noise>"
             << endl;
        cout << "   ========================================================="
             << endl;
        cout << "      <input file>  - *.ppm file with input colors" << endl;
        cout << "                      --OR--" << endl;
        cout << "                      *.lightning file from a previous run"
             << endl;
        cout << "      <output file> - The OpenEXR file to output" << endl;
        cout << "      <scale>       - Scaling constant for final image."
             << endl;
        cout << "   Press 'q' to terminate the simulation prematurely." << endl;
        cout << "   Send questions and comments to kim@cs.unc.edu" << endl;
        return 1;
    }

    if (argv[4][0] == '1') {
        include_noise = true;
    }

    cout << endl
         << "Lumos: A lightning generator v0.1" << endl;
    cout << "------------------------------------------------------" << endl;

    // store the input params
    inputFile = string(argv[1]);
    outputFile = string(argv[2]);
    if (argc > 3) scale = atoi(argv[3]);

    // see if the input is a *.lightning file
    if (inputFile.size() > 10) {
        string postfix =
            inputFile.substr(inputFile.size() - 9, inputFile.size());

        cout << " Using intermediate file " << inputFile << endl;
        if (postfix == string("lightning")) {
            potential->readDAG(inputFile.c_str());
            renderGlow(outputFile, scale);
            delete potential;
            return 0;
        }
    }

    // read in the *.ppm input file
    if (!loadImages(inputFile)) {
        cout << " ERROR: " << inputFile.c_str() << " is not a valid PPM file."
             << endl;
        return 1;
    }
    cout << " " << inputFile << " read." << endl
         << endl;

    // loop simulation until it hits a terminator
    cout << " Total particles added: ";
    glutInit(&argc, argv);
    glutMain();

    return 0;
}
#else
#include <fstream>
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input file> <output file> <N>" << std::endl;
        return 1;
    }

    return do_convolve(argv[1], argv[2], atoi(argv[3]));
}
#endif
