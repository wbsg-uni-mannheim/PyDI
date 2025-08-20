/*
 * Copyright (c) 2017 Data and Web Science Group, University of Mannheim, Germany (http://dws.informatik.uni-mannheim.de/)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

package de.uni_mannheim.informatik.dws.winter.preprocessing.datatypes;

import java.text.ParseException;

import org.slf4j.Logger;

import de.uni_mannheim.informatik.dws.winter.preprocessing.units.Quantity;
import de.uni_mannheim.informatik.dws.winter.preprocessing.units.Unit;
import de.uni_mannheim.informatik.dws.winter.preprocessing.units.UnitCategory;
import de.uni_mannheim.informatik.dws.winter.preprocessing.units.UnitCategoryParser;
import de.uni_mannheim.informatik.dws.winter.utils.WinterLogManager;

public class ValueNormalizer {
	
	private static final Logger logger = WinterLogManager.getLogger();
	
	/**
	 * Converts a String into the given type
	 * @param value
	 * @param type
	 * @param unitcategory
	 * @return an object representing the value in the corresponding data type
	 */
    public Object normalize(String value, DataType type, UnitCategory unitcategory) {
        Object normalizedValue = null;
        
        if(value!=null) {
	        try {
		        switch (type) {
		            case string:
		            	normalizedValue = value;
		                break;
		            case date:
		            	normalizedValue = DateJavaTime.parse(value);
		                break;
		            case numeric:
		                //TODO: how to handle numbers with commas (German style)
		      
		            	Quantity quantity = UnitCategoryParser.checkQuantity(value);
		                Unit unit = UnitCategoryParser.checkUnit(value, unitcategory);
		                
		                normalizedValue = UnitCategoryParser.transform(value, unit, quantity); 
		                
		                break;
		            case bool:
		            	normalizedValue = Boolean.parseBoolean(value);
		                break;
		            case coordinate:
		            	normalizedValue = value;
		                break;
		            case link:
		            	normalizedValue = value;
		            default:
		                break;
		        }
	        } catch(ParseException e) {
	        	logger.trace("ParseException for " + type.name() + " value: " + value);
	        		//e.printStackTrace();
	        }
        }
        
        return normalizedValue;
    }
}
