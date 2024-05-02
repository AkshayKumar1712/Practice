import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class TimeMap {

    
    class InnerTimeMap {
        String value;
        int timestamp;
        InnerTimeMap(String value, int timestamp) {
            this.value = value;
            this.timestamp = timestamp;
        }
    }

    Map<String, List<InnerTimeMap>> mapToStoreKeyValueTimeStamps;

    public TimeMap() {
        mapToStoreKeyValueTimeStamps = new HashMap<>();
    }
    
    public void set(String key, String value, int timestamp) {
        InnerTimeMap innerTimeMapObj = new InnerTimeMap(value, timestamp);
        if(mapToStoreKeyValueTimeStamps.containsKey(key)) {
            mapToStoreKeyValueTimeStamps.get(key).add(innerTimeMapObj);
        }
        else{
            List<InnerTimeMap> valueTimeStampList = new ArrayList<>();
            valueTimeStampList.add(innerTimeMapObj);
            mapToStoreKeyValueTimeStamps.put(key, valueTimeStampList);
        }
    }
    
    public String get(String key, int timestamp) {
        List<InnerTimeMap> valueTimeStampList = mapToStoreKeyValueTimeStamps.get(key);
        if(valueTimeStampList != null) {
            int high = valueTimeStampList.size() - 1, low = 0;
            String result = "";
            while (low <= high) {
                int mid = low + ((high-low)/2);
                InnerTimeMap valueTimeStampObj = valueTimeStampList.get(mid);
                if(valueTimeStampObj.timestamp == timestamp)
                return valueTimeStampObj.value;

                else if(valueTimeStampObj.timestamp < timestamp)
                {
                    low = mid + 1;
                    result = valueTimeStampObj.value;
                }

                else{
                    high = mid - 1;
                }
            }
            return result;
        }
        return null;
    }
    
    public static void main(String[] args) {
        TimeMap timeMap = new TimeMap();
        timeMap.set("foo", "bar", 1);
        System.out.println(timeMap.get("foo", 1));
        System.out.println(timeMap.get("foo", 3));
        timeMap.set("foo", "bar2", 4);
        System.out.println(timeMap.get("foo", 4));
        System.out.println(timeMap.get("foo", 5));
    }
}