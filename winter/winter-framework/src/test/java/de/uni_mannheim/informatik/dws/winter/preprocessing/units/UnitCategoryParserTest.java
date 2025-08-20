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
package de.uni_mannheim.informatik.dws.winter.preprocessing.units;


import org.junit.Test;
import junit.framework.TestCase;

public class UnitCategoryParserTest extends TestCase {

    @Test
    public void testCheckUnit() {
    	
    	//check Unit 
        assertNull(UnitCategoryParser.checkUnit("100 km", null));
        assertNotNull(UnitCategoryParser.checkUnit("100 km", UnitCategoryParser.getUnitCategory("All")));
        assertNull(UnitCategoryParser.checkUnit("100 km", UnitCategoryParser.getUnitCategory("Area")));
        assertNotNull(UnitCategoryParser.checkUnit("100 km", UnitCategoryParser.getUnitCategory("Distance")));
        
        // check Quantifier
        assertNull(UnitCategoryParser.checkQuantity("100 km"));
        assertNull(UnitCategoryParser.checkUnit("100 million", null));
        assertNotNull(UnitCategoryParser.checkQuantity("100 million"));
        assertEquals(UnitCategoryParser.getQuantity("million"),UnitCategoryParser.checkQuantity("100 million km"));
        
     // check Unit
        assertEquals("kilometre",UnitCategoryParser.checkUnit("100 million km", UnitCategoryParser.getUnitCategory("Distance")).getName());
        assertEquals(UnitCategoryParser.checkQuantity("million"), UnitCategoryParser.checkQuantity("100 million km"));
        
     // check Unit
        assertNull(UnitCategoryParser.checkUnit("$ 50", null));
        assertNull(UnitCategoryParser.checkUnit("$ 50 million", null));
    }

}