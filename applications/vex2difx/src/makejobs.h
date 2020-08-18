/***************************************************************************
 *   Copyright (C) 2015 by Walter Brisken                                  *
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
 * $Id: makejobs.h 6894 2015-07-28 16:19:06Z WalterBrisken $
 * $HeadURL: https://svn.atnf.csiro.au/difx/applications/vex2difx/trunk/src/util.h $
 * $LastChangedRevision: 6894 $
 * $Author: WalterBrisken $
 * $LastChangedDate: 2015-07-29 02:19:06 +1000 (Wed, 29 Jul 2015) $
 *
 *==========================================================================*/

#ifndef __MAKEJOBS__H__
#define __MAKEJOBS__H__

#include <list>
#include <utility>
#include <vector>
#include <string>
#include "vex_data.h"
#include "job.h"
#include "corrparams.h"

void makeJobs(std::vector<Job>& J, const VexData *V, const CorrParams *P, std::list<Event> &events, std::list<std::pair<int,std::string> > &removedAntennas, int verbose);

#endif
