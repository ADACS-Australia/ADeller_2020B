# -*- coding: utf-8 -*-

#===========================================================================
# Copyright (C) 2016  Max-Planck-Institut für Radioastronomie, Bonn, Germany
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#===========================================================================
# SVN properties (DO NOT CHANGE)
#
# $Id: slotaction.py 7658 2017-02-22 14:53:57Z HelgeRottmann $
# $HeadURL: https://svn.atnf.csiro.au/difx/libraries/python/trunk/difxdb/business/slotaction.py $
# $LastChangedRevision: 7658 $
# $Author: HelgeRottmann $
# $LastChangedDate: 2017-02-23 01:53:57 +1100 (Thu, 23 Feb 2017) $
#
#============================================================================
from difxdb.model import model

def slotExists(session, location):
    
    if (session.query(model.Slot).filter_by(location=location).count() > 0):
        return(True)
    else:
        return(False)
    
def getSlotById(session, id):
    
    return(session.query(model.Slot).filter_by(id=id).one())

def getSlotByLocation(session, location):
    
    return(session.query(model.Slot).filter_by(location=location).one())
    
def getOccupiedSlots(session):
    '''
    Returns a collection of Slot objects that are currently occupied by modules 
    '''
    
    return(session.query(model.Slot).filter(model.Slot.moduleID != None).order_by(model.Slot.location).all())

def getEmptySlots(session):   
    '''
    Returns a collection of Slot objects that are currently not occupied by modules 
    '''
    
    result =  session.query(model.Slot).order_by(model.Slot.location).filter_by(isActive = 1).order_by(model.Slot.location).filter(model.Slot.moduleID == None)
    
    return(result)

def getAllSlots(session):   
    '''
    Returns all slots
    '''
    
    result =  session.query(model.Slot).order_by(model.Slot.location).filter_by(isActive = 1)
    return(result)