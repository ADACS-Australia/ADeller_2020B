/***************************************************************************
 *   Copyright (C) 2008-2023 by Walter Brisken                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
//===========================================================================
// SVN properties (DO NOT CHANGE)
//
// $Id: sniffer.h 10952 2023-04-07 18:12:10Z WalterBrisken $
// $HeadURL: https://svn.atnf.csiro.au/difx/applications/difx2fits/trunk/src/sniffer.h $
// $LastChangedRevision: 10952 $
// $Author: WalterBrisken $
// $LastChangedDate: 2023-04-08 04:12:10 +1000 (Sat, 08 Apr 2023) $
//
//============================================================================
#ifndef __SNIFFER_H__
#define __SNIFFER_H__

#include <stdio.h>
#include <complex.h>
#include <fftw3.h>
#include "fitsUV.h"

#define DEFAULT_MAX_SNIFFER_MEMORY 2000000000LL

struct _Sniffer;

typedef struct _Sniffer Sniffer;

Sniffer *newSniffer(const DifxInput *D, int nComplex, const char *filebase, double solint, int writeBandpass);

void deleteSniffer(Sniffer *S);

int feedSnifferFITS(Sniffer *S, const DifxVis *dv);

long long getSnifferMemoryUsage(const Sniffer *S);

#endif
