/***************************************************************************
 *   Copyright (C) 2015-2021 by Walter Brisken & Adam Deller               *
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
/*===========================================================================
 * SVN properties (DO NOT CHANGE)
 *
 * $Id: vex_clock.cpp 10363 2022-01-27 22:57:59Z WalterBrisken $
 * $HeadURL: https://svn.atnf.csiro.au/difx/applications/vex2difx/branches/multidatastream_refactor/src/vex2difx.cpp $
 * $LastChangedRevision: 10363 $
 * $Author: WalterBrisken $
 * $LastChangedDate: 2022-01-28 09:57:59 +1100 (Fri, 28 Jan 2022) $
 *
 *==========================================================================*/

#include "vex_clock.h"

std::ostream& operator << (std::ostream &os, const VexClock &x)
{
	os << "Clock(" << x.mjdStart << ": " << x.offset << ", " << x.rate;
	if(x.accel != 0.0 || x.jerk != 0.0)
	{
		os << ", " << x.accel;
	}
	if(x.jerk != 0.0)
	{
		os << ", " << x.jerk;
	}
	
	os << ", " << x.offset_epoch << ")";

	return os;
}
