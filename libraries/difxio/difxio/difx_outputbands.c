/***************************************************************************
 *   Copyright (C) 2008-2015 by Walter Brisken                             *
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
// $Id: difx_outputbands.c 10579 2022-08-02 10:58:00Z JanWagner $
// $HeadURL:  $
// $LastChangedRevision: 10579 $
// $Author: JanWagner $
// $LastChangedDate: 2022-08-02 20:58:00 +1000 (Tue, 02 Aug 2022) $
//
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "difxio/difx_input.h"


const char outputBandwidthModeNames[][MAX_OUTPUT_BANDWIDTH_MODE_NAME_LENGTH] =
{
	"OFF",
	"AUTO",
	"USER"
	"UNKNOWN"
};


enum OutputBandwidthMode stringToOutputBandwidthMode(const char *str)
{
	enum OutputBandwidthMode t;

	for(t = 0; t < NumOutputBandwidthModes; ++t)
	{
		if(strcmp(str, outputBandwidthModeNames[t]) == 0)
		{
			break;
		}
	}

	return t;
}
