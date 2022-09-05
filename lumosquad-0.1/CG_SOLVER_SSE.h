///////////////////////////////////////////////////////////////////////////////////
// File : CG_SOLVER_SSE.h
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

#ifndef CG_SOLVER_SSE_H
#define CG_SOLVER_SSE_H

#include "CG_SOLVER.h"
#include <xmmintrin.h>

////////////////////////////////////////////////////////////////////
/// \brief Conjugate gradient Poisson solver that exploits SSE.
////////////////////////////////////////////////////////////////////
class CG_SOLVER_SSE : public CG_SOLVER
{
public:
  //! constructor
  CG_SOLVER_SSE(int maxDepth, int iterations = 10, int digits = 1);
  //! destructor
  ~CG_SOLVER_SSE();

  //! solve the Poisson problem using SSE
  virtual int solve(list<CELL*> cells);
  
private:
  //! reallocate the SSE-friendly scratch arrays
  virtual void reallocate();
  
  // SSE linear algebra operators
  inline float dotSSE(float* x, float* y);
  inline void saxpySSE(float a, float* x, float* y);
  inline void saypxSSE(float a, float* x, float* y);
  inline float maxSSE(float* x);
  inline void addSSE(float* x, float* y);
  inline void multiplySSE(float* x, float* y);
  inline void multiplySSE(float* x, float* y, float* z);
  inline void multiplySubtractSSE(float* w, float* x, float* y, float* z);
  inline void setSSE(float* x, float val);
  inline void wipeSSE(float* x);
  inline void copySSE(float* x, float* y);
};

#endif
