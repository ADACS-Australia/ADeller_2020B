/***************************************************************************
 *   Copyright (C) 2016-2017 by Walter Brisken                             *
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
// $Id: utils.h 7763 2017-05-16 18:18:27Z WalterBrisken $
// $HeadURL: $
// $LastChangedRevision: 7763 $
// $Author: WalterBrisken $
// $LastChangedDate: 2017-05-17 04:18:27 +1000 (Wed, 17 May 2017) $
//
//============================================================================

#ifndef __UTILS_H__
#define __UTILS_H__

#include <cstdio>

char *fgetsNoCR(char *line, int MaxLineLength, FILE *in);

#endif
