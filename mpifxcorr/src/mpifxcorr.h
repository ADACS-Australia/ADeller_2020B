/***************************************************************************
 *   Copyright (C) 2006-2016 by Adam Deller                                *
 *                                                                         *
 *   This program is free software: you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation, either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 ***************************************************************************/
//===========================================================================
// SVN properties (DO NOT CHANGE)
//
// $Id: mpifxcorr.h 8437 2018-09-07 10:55:05Z JanWagner $
// $HeadURL: https://svn.atnf.csiro.au/difx/mpifxcorr/trunk/src/mpifxcorr.h $
// $LastChangedRevision: 8437 $
// $Author: JanWagner $
// $LastChangedDate: 2018-09-07 20:55:05 +1000 (Fri, 07 Sep 2018) $
//
//============================================================================
#ifndef MPIFXCORR_H
#define MPIFXCORR_H

#include <string>

///constants for MPI ids
namespace fxcorr
{
  static const int MANAGERID = 0;
  static const int FIRSTTELESCOPEID = 1;
}

static const int FLAGS_PER_INT = 30;

#endif
// vim: shiftwidth=2:softtabstop=2:expandtab
