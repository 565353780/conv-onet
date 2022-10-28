#!/usr/bin/env python
# -*- coding: utf-8 -*-

from conv_onet.Data.field.field import Field


class IndexField(Field):
    ''' Basic index field.'''

    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, filed):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        return True
