import java.util.*;

public class Test {

    class PairInt{
        int r;
        int c;
        public PairInt(int r, int c) {
            this.r = r;
            this.c = c;
        }
    }

    private boolean subsetSumRec(int[] arr, int sum, int index){
        if(sum == 0)
        return true;

        if(sum > 0 && index >= arr.length)
        return false;

        return  (arr[index] <= sum) ? subsetSumRec(arr, sum - arr[index], index+1) : false || subsetSumRec(arr, sum, index+1);
    }

    public boolean subsetSum(int[] arr, int sum){
        return subsetSumRec(arr, sum, 0);
    }

    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> prefixSumMap = new HashMap<>();
        prefixSumMap.put(0, 1);
        int prefixSum = 0, result = 0;
        for (int i = 0; i < nums.length; i++) {
            prefixSum += nums[i];
            result += prefixSumMap.getOrDefault(prefixSum - k, 0);
            prefixSumMap.put(prefixSum, prefixSumMap.getOrDefault(prefixSum, 0) + 1);
        }
        return result;
    }

    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1;
        int maxArea = -1;
        while (left < right) {
            maxArea = Integer.max(Integer.min(height[left], height[right]) * (right - left), maxArea);
            if(height[left] < height[right]) 
            left++;
            else
            right--;
        } 
        return maxArea;
    }

    public static double squareRoot(double number, double epsilon) {
        if (number < 0) {
            throw new IllegalArgumentException("Cannot compute square root of a negative number");
        }

        double guess = number / 2; // Initial guess can be arbitrary, but this is a common starting point

        while (Math.abs(guess * guess - number) > epsilon) {
            guess = (guess + number / guess) / 2;
        }

        return guess;
    }

    public int findFloorInSortedArray(int[] arr, int key) {
        int start = 0, end = arr.length - 1;
        int result = 0;
        while (start <= end) {
            int mid = start + ((end-start)/2);
            if(arr[mid] == key)
            return key;

            if(arr[mid] < key)
            {
                result = arr[mid];
                start = mid + 1;
            }
            else
            end = mid-1;
        }
        return result;
    }

    // This is similar to the ceil in a sorted array
    public char findNextAvailCharInArray(char[] arr, char key) {
        int start = 0, end = arr.length - 1;
        char result = ' ';
        while (start <= end) {
            int mid = start + ((end-start)/2);
            
            if(arr[mid] == key)
            return key;

            if(arr[mid] > key)
            {
                result = arr[mid];
                end = mid - 1;
            }
            else
            start = mid + 1;
        }
        return result;
    }

    // This can be used for the maximum in a bitonic array
    public int findPeakElement(int[] arr) {
        int start = 0, end = arr.length - 1;
        while (start <= end) {
            int mid = start + ((end-start)/2);
            
            if(mid + 1 <= arr.length - 1 && mid - 1 >= 0 && (arr[mid-1] < arr[mid] && arr[mid+1] < arr[mid]))
            return arr[mid];

            if(mid-1 < 0 && arr[mid+1] < arr[mid])
            return arr[mid];

            if(mid+1 > arr.length - 1 && arr[mid-1] < arr[mid])
            return arr[mid];

            else if(arr[mid+1] > arr[mid])
            start = mid+1;

            else
            end = mid - 1;
        }

        return arr[start];
    }

    public int binarySearch(int[] arr, int start, int end, int key) {
        while (start <= end) {
            int mid = start + ((end-start)/2);
            if(arr[mid] == key)
            return mid;

            else if(arr[mid] < key)
            start = mid + 1;

            else
            end = mid-1;
        }
        return -1;
    }

    public int findMinInRotatedSortedArray(int[] nums) {
        if(nums.length == 1)
        return nums[0];

        int low = 0, high = nums.length - 1;
        while (low <= high) {
            int mid = low + ((high-low)/2);

            if(nums[low] < nums[high])
            return nums[low];

            if(mid - 1 >= 0 && mid + 1 < nums.length && nums[mid] < nums[mid-1] && nums[mid] < nums[mid+1])
            return nums[mid];

            if(mid-1 < 0 &&  nums[mid] < nums[mid+1])
            return nums[mid];

            if(mid+1 >= nums.length && nums[mid] < nums[mid-1])
            return nums[mid];

            else if(nums[mid] > nums[high])
            low = mid + 1;

            else
            high = mid - 1;
        }
        return nums[low];
    }

    public int longestConsecutive(int[] nums) {
        Set<Integer> numSet = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            numSet.add(nums[i]);
        }
        
        int result = 0;
        for (Integer num : nums) {
            if(!numSet.contains(num - 1)) {
                int seqLength = 1;
                while (numSet.contains(++num)) {
                    seqLength++;
                }
                result = Integer.max(seqLength, result);
            }
            if(result > nums.length/2)
            break;
        }
        return result;
    }

    public int searchInRotatedArray(int[] nums, int target) {
        int low = 0, high = nums.length - 1;
        int n = nums.length;
        int minIndex = 0;
        
        if(n == -1)
        return nums[low] == target ? low : -1;

        while (low <= high) {
            int mid = low + ((high-low)/2);
            int midRight = (mid + 1) % n;
            int midLeft = (mid-1 + n) % n;
            if(nums[mid] < nums[midRight] && nums[mid] < nums[midLeft])
            {
                minIndex = mid;
                break;
            }

            else if(nums[low]<=nums[mid])
            low = midRight;

            else
            high = midLeft;
        }

        if(nums[minIndex] == target)
        return minIndex;

        int leftSortedArraySearch = binarySearch(nums, 0, minIndex - 1, target);
        
        return leftSortedArraySearch != -1 ? leftSortedArraySearch : binarySearch(nums, minIndex + 1, n - 1, target);
    }

    public int averageGrade(String[][] grades) {
        double result = 0;
        Map<String, List<Integer>> gradesMap = new HashMap<>();
        for (int i = 0; i < grades.length; i++) {
            if(grades[i].length != 2)
            return -1;

            String studentName = grades[i][0];
            int studentGrade = Integer.parseInt(grades[i][1]);

            if(gradesMap.containsKey(studentName)) {
                gradesMap.get(studentName).add(studentGrade);
            }

            else{
                List<Integer> newStudentGradeList = new ArrayList<>();
                newStudentGradeList.add(studentGrade);
                gradesMap.put(studentName, newStudentGradeList);
            }
        }

        for (String student: gradesMap.keySet()) {
            double avg = gradesMap.get(student).stream().mapToInt(a -> a).average().orElse(0.0);
            result = Double.max(avg, result);
        }
        
        return (int)Math.floor(result);
    }

    public int snowPack(int[] height) {
        int left = 0, right = height.length - 1;
        int leftMax = 0, rightMax = 0;
        int trappedSnow = 0;
        while(left <= right) {
            // This means the left pointer will be moved and result will be updated 
            // based on the leftMax
            if(height[left] <= height[right]) {
                // If the current height is more than the maxima 
                // then update the maxima and no water can be store as there is no bound on the left
                if(height[left] >= leftMax)
                leftMax = height[left];

                // Since there is a maxima to the left some water can be stored
                // No need to worry about right because this point is reached because on the right there is
                // something greater than the leftMax;
                else
                trappedSnow += leftMax - height[left];

                left++;
            }

            // Same explanation but right instead of left
            else {
                if(height[right] >= rightMax)
                rightMax = height[right];

                else
                trappedSnow += rightMax - height[right];

                right--;
            }
        }

        return trappedSnow;
    }
    
    public int optimalPathRecDp(int[][] grid, int i, int j, int[][] dp) {

        if(i == grid.length - 1 && j == grid[0].length - 1)
        return grid[i][j];

        if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length)
        return 0;

        // Need to check with the interviewer if input could have negative numbers, if yes then fill the array with INT_MIN
        if(dp[i][j] != -1)
        return dp[i][j];

        return dp[i][j] = grid[i][j] + Integer.max(optimalPathRecDp(grid,i,j+1,dp), optimalPathRecDp(grid,i-1,j,dp));
    }

    public int optimalPath(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }

        int m = grid.length;
        int n = grid[0].length;

        for (int i = m - 1; i >= 0; i--) {
            for (int j = 0; j < n; j++) {
                if(i < m - 1 && j > 0) {
                    grid[i][j] = Math.max(grid[i+1][j], grid[i][j-1]);
                }

                else if(i < m - 1)
                {
                    grid[i][j] += grid[i+1][j];
                }

                else if(j > 0) {
                    grid[i][j] += grid[i][j-1];
                }
            }
        }

        // The optimal path sum will be stored at the top-right corner of the grid array
        return grid[0][n - 1];
    }

    public double medianTwoSortedArrays(int arr1[], int arr2[]) {
        double median = -1.0;
        int n1 = arr1.length, n2 = arr2.length;
        
        // It is optimal to do the BS on the smallest array
        if(n1 > n2)
        return medianTwoSortedArrays(arr2, arr1);
        
        int low = 0, high = n1;
        int leftHalfSize = (n1 + n2 + 1)/2;
        while (low <= high) {
            int mid1 = (low + high)/2;
            int mid2 = leftHalfSize - mid1;

            int l1 = Integer.MIN_VALUE, l2 = Integer.MIN_VALUE;
            int r1 = Integer.MAX_VALUE, r2 = Integer.MAX_VALUE;

            if(mid1 - 1 >= 0)
            l1 = arr1[mid1 - 1];
            
            if(mid2 - 1 >= 0)
            l2 = arr2[mid2 - 1];
            
            if(mid1 < n1)
            r1 = arr1[mid1];

            if(mid2 < n2)
            r2 = arr2[mid2];

            // Split is perfect return the median
            if(l1 <= r2 && l2 <= r1)
            {
                if((n1+n2)%2 == 1)
                return Integer.max(l1, l2)/1.0;

                return (Integer.max(l1, l2) + Integer.min(r1, r2))/2.0;
            }

            // This means less elements are considered from the array 2 than required
            else if(l1 > r2)
            high = mid1 - 1;
            
            else
            low = mid1 + 1;
        }

        return median;
    }

    public String decimalConverstion(int numerator, int denominator) {
        if (numerator == 0) {
            return "0";
        }
        StringBuilder res = new StringBuilder();
        // "+" or "-"
        res.append(((numerator > 0) ^ (denominator > 0)) ? "-" : "");
        long num = Math.abs((long)numerator);
        long den = Math.abs((long)denominator);
        
        // integral part
        res.append(num / den);
        num %= den;
        if (num == 0) {
            return res.toString();
        }
        
        // fractional part
        res.append(".");
        HashMap<Long, Integer> map = new HashMap<Long, Integer>();
        map.put(num, res.length());
        while (num != 0) {
            num *= 10;
            res.append(num / den);
            num %= den;
            if (map.containsKey(num)) {
                int index = map.get(num);
                res.insert(index, "(");
                res.append(")");
                break;
            }
            else {
                map.put(num, res.length());
            }
        }
        return res.toString();
    }

    // BFS to get the shortest path and then a hashmap that keeps track of child and their parents to back track and print the path
    public String trainMap(Map<String, List<String>> graph, String fromStationName, String toStationName) {
        if(!graph.containsKey(fromStationName))
        return "NO PATH FOUND!";

        Queue<String> queueForBFS = new LinkedList<>();
        Set<String> visitedStations = new HashSet<>();
        Map<String, String> childToParentMap = new HashMap<>();
        queueForBFS.add(fromStationName);
        while (!queueForBFS.isEmpty()) {
            String currStation = queueForBFS.poll();
            visitedStations.add(currStation);
            
            if(currStation.equals(toStationName))
            {
                List<String> path = new ArrayList<>();
                path.add(toStationName);
                String parent = "", backtrackHelper = toStationName;
                while (!parent.equals(fromStationName)) {
                    parent = childToParentMap.get(backtrackHelper);
                    path.add(parent);
                    backtrackHelper = parent;
                }
                Collections.reverse(path);
                return String.join(" -> ", path);
            }
            List<String> children = graph.getOrDefault(currStation, new ArrayList<>());
            for (String child : children) {
                if(!visitedStations.contains(child)){
                    childToParentMap.putIfAbsent(child, currStation);
                    queueForBFS.add(child);
                }
            }
        }

        return "NO PATH FOUND!";
    }

    // Need to get the optimal solution
    public int longestTree(Map<Integer, Integer> forest) {
        Map<Integer, Integer> rootLengthsMap = new HashMap<>();
        for (Integer child : forest.keySet()) {
            int immediateParent = forest.get(child);
            int root = getRoot(forest, immediateParent);
            rootLengthsMap.put(root, rootLengthsMap.getOrDefault(rootLengthsMap, 0) + 1);
        }

        int maxChild = -1, resultTreeId = Integer.MAX_VALUE;

        for (Integer root : rootLengthsMap.keySet()) {
            if(maxChild <= rootLengthsMap.get(root))
            {
                maxChild = rootLengthsMap.get(root);
                resultTreeId = (resultTreeId > root) ? root : resultTreeId;
            }
        }
        return resultTreeId;
    }

    private int getRoot(Map<Integer, Integer> forest, int immediateParent) {
        int root = immediateParent;
        while (forest.containsKey(immediateParent)) {
            root = forest.get(immediateParent);
            immediateParent = root;
        }
        return root;
    }

    /*
     * EASY QUESTIONS BELOW!
     */

    // TC - O(N)
    public int getSecondMinimumElement(int[] arr){
        int firstMin = Integer.MAX_VALUE, secondMin = Integer.MAX_VALUE;
        for (int i = 0; i < arr.length; i++) {
            if(arr[i] <= firstMin)
            {
                secondMin = firstMin;
                firstMin = arr[i];
            }
            else if(arr[i] > firstMin && arr[i] <= secondMin)
            secondMin = arr[i];
        }
        return secondMin;
    }

    // TC - O(logb - base 2)
    public long powerOfANumber(int a, int b) {
        long result = 1;
        while (b > 0) {
            if(b % 2 == 1)
            result = result * a;
            b >>= 1;
            a *= a;
        }   
        return result;
    }

    // TC -> O(input.length()) + O(26)
    public int firstNonRepeatingChar(String input){
        Map<Character, Integer> charFreq = new HashMap<>(); 
        
        for (int i = 0; i < input.length(); i++) {
            char currChar = input.charAt(i);
            if(charFreq.containsKey(currChar)) {
                charFreq.put(currChar, -2);
            }
            else
                charFreq.put(currChar, i);
        }
        int result = Integer.MAX_VALUE;
        for (Character key: charFreq.keySet()) {
            int index = charFreq.get(key);
            if(index != -2)
            result = Integer.min(result, index);
        }
        return result == Integer.MAX_VALUE ? -1 : result;
    }

    // TC - O(N) + O(N)
    public int pivotIndexForArraySum(int arr[]) {
        int leftSum = 0, totalSum = 0;
        for (int i = 0; i < arr.length; i++) {
            totalSum += arr[i];
        }
        for (int i = 0; i < arr.length; i++) {
            int rightSum = totalSum - arr[i] -leftSum; 
            if(leftSum == rightSum)
            return i;
            leftSum += arr[i];
        }
        return -1;
    }

    // Magic Potion
    // Complexity -> O(N*TC(substring method))
    private static int minimalSteps(String ingredients) {
        int n = ingredients.length();
        int[] dp = new int[n];
        dp[0] = 1;
        for (int i = 1; i < n; i++) {
            // Check repeat only for even number of characters (odd index)
            if (i % 2 == 1 && (repeatString(ingredients, 0, i / 2, i))) {
                // If repeated use the length of the first block + 1 (for *)
                dp[i] = dp[i / 2] + 1;
            } else {
                // If not just increment
                dp[i] = dp[i-1] + 1;
            }
        }
        return dp[n-1];
    }

    private static boolean repeatString(String s, int i, int j, int k) {
        return s.substring(i, j+1).equals(s.substring(j+1, k+1));
    }

    int solve(String ingredients, int index){ // index = n-1
        int n = ingredients.length();
        int i = 0;
        if(i==0){return 1;}
     
        // Check repeat only for even number of characters (odd index)
        if (i % 2 == 1 && (repeatString(ingredients, 0, i / 2, i))) {
        // If repeated use the length of the first block + 1 (for *)
            return solve(ingredients, i / 2) + 1;
        } else {
            // If not just increment
            return solve(ingredients, i-1) + 1;
        }
    }

    public int lengthOfLISHelper(int[] nums, int index, int prevIndex) {
        if(index > nums.length - 1)
        return 0;
        
        if(prevIndex != -1 && nums[index] < nums[prevIndex]) {
            return lengthOfLISHelper(nums, index + 1, prevIndex);
        }

        return Integer.max(1+lengthOfLISHelper(nums, index + 1, index), lengthOfLISHelper(nums, index + 1, prevIndex));
    }

    public int[] productExceptSelf(int[] nums) {
        int[] output = new int[nums.length];
        int[] prefix = new int[nums.length];
        int[] postfix = new int[nums.length];
        int n = nums.length;
        int temp = 1;
        for(int i = 0; i < n; i++) {
            temp *= nums[i];
            prefix[i] = temp;
        }
        temp = 1;
        for(int i = n-1; i >= 0; i--) {
            temp *= nums[i];
            postfix[i] = temp;
        }

        for(int i = 0; i < n; i++) {
            if(i == 0)
            output[i] = postfix[i+1];

            else if(i==n-1)
            output[i] = prefix[i-1];

            else
            output[i] = prefix[i-1] * postfix[i+1];
        }

        return output;
    }

    public int[] productExceptSelfOptimised(int[] nums) {
        int[] output = new int[nums.length];
        int n = nums.length;
        int temp = 1;
        output[0] = 1;
        for(int i = 0; i < n - 1; i++) {
            temp *= nums[i];
            output[i + 1] = temp;
        }
        temp = 1;
        for(int i = n-1; i > 0; i--) {
            temp *= nums[i];
            output[i - 1] *= temp;
        }

        return output;
    }


    public int[] longestUniformSubString(String s) 
    {
        int maxFreq = -1, startIndex = 0;
        char prevChar = s.charAt(0);
        int charCount = 1;
        for (int i = 1; i < s.length(); i++) {
            if (prevChar == s.charAt(i)) {
                charCount++;
            }
            else {
                prevChar = s.charAt(i);
                charCount = 1;
            }
            if(charCount > maxFreq)
            {
                maxFreq = charCount;
                startIndex =  i - maxFreq + 1;
            }
        }
        return new int[] {maxFreq, startIndex};
    }

    // TC - O(sqrt(N))
    public List<Integer> primeFactors(int n) {
        List<Integer> result = new ArrayList<>();
        while (n%2 == 0) {
            n /= 2;
            result.add(2);
        }
        // Why sqrt(n) because n = a * b and at max both a = b = sqrt(n)
        for (int i = 3; i <= Math.sqrt(n); i+=2) {
            while (n%i == 0) {
                n /= i;
                result.add(i);
            }
        }

        // What if n is a prime number, eg 17
        if(n > 2) 
        result.add(n);

        return result;
    }

    // A number in pascal triangle comes from the sum of top row element at it's position and it's previous position therefore ncr
    public long pascalTriangleNumber(int n, int k) {
        if(n == 0 || k == n)
        return 1;

        if(k > n - k)
            k = n - k;

        long result = 1;
        for (int i = 0; i < k; i++) {
            result *= (n-i);
            result /= (i+1);
        }

        return result;
    }

    // O(N) 
    public int election(int n, int k)
    {
        if(n == k)
        return -1;

        LinkedList<Integer> studentsList = new LinkedList<>();
        for (int i = 1; i <= n; i++) {
            studentsList.add(i);
        }

        int j = 0;
        while (studentsList.size() > 1) {
            int studentSongStopIndex = (j+(k -1))%studentsList.size();
            studentsList.remove(studentSongStopIndex);
            j = (j + (k -1)) % (studentsList.size() + 1);
        }
        
        return studentsList.getFirst();
    }

    public int atoi(String num) {
        if(num.length() == 0)
        return 0;
        int result = 0;
        int multiplier = 1;
        for (int i = 0; i < num.length(); i++) {
            if(i==0 && num.charAt(i) == '-')
            {
                multiplier = -1;
                continue;
            }
            
            if(num.charAt(i) < '0' || num.charAt(i) > '9')
            {
                break;
            }
            result = result * 10 + num.charAt(i) - '0';
        }
        return multiplier * result;
    }

    /*
        https://leetcode.com/discuss/interview-question/4668551/SnapChat-Tech-Screen-Feb-2024
        Have to place pairs in a way such that second character in each pair is same as first character in next pair. Like :
        [Ba, _a, cb, dc] to[ _a, ab, bc, cd]
        
        Input : [Ba, _a, cb, dc]
        output : _abcd

        Input : [El, ll, lo, _h, he]
        output : _hello
     */
    // public String connectPairs(String[] input) {
        
    // }

    public int[][] shortestPathBinaryMatrix(int[][] input) {
        List<Integer> listOf1sInRow1 = new ArrayList<>();
        for (int j = 0; j < input[0].length; j++) {
            if(input[0][j] == 1 && input[1][j] == 1)
            listOf1sInRow1.add(j);
        }

        Stack<PairInt> indexesForShortestPath = new Stack<>();
        int currMin = Integer.MAX_VALUE;
        for (int startingIndex : listOf1sInRow1) {
            boolean[][] visited = new boolean[input.length][input[0].length];
            Queue<PairInt> queueForBFS = new LinkedList<>();
            queueForBFS.add(new PairInt(0, startingIndex));
            Stack<PairInt> indexesForPath = new Stack<>();
            while (!queueForBFS.isEmpty()) {
                PairInt currentPair = queueForBFS.poll();
                indexesForPath.push(currentPair);
                visited[currentPair.r][currentPair.c] = true;
                if(currentPair.r == input.length - 1)
                break;

                PairInt rightPair = new PairInt(currentPair.r, currentPair.c + 1);
                PairInt leftPair = new PairInt(currentPair.r, currentPair.c - 1);
                PairInt topPair = new PairInt(currentPair.r + 1, currentPair.c);

                if(currentPair.r > 0) {
                    if(currentPair.c > 0 && currentPair.c < input[0].length - 1){
                        if(input[rightPair.r][rightPair.c] == 1 && !visited[rightPair.r][rightPair.c])
                        queueForBFS.add(rightPair);
                        if(input[leftPair.r][leftPair.c] == 1 && !visited[leftPair.r][leftPair.c])
                        queueForBFS.add(leftPair);
                    }
                    
                    else if(currentPair.c == 0 && input[rightPair.r][rightPair.c] == 1 && !visited[rightPair.r][rightPair.c]) {
                        queueForBFS.add(rightPair);
                    }

                    else if(currentPair.c == input[0].length - 1 && input[leftPair.r][leftPair.c] == 1 && !visited[leftPair.r][leftPair.c]) {
                        queueForBFS.add(leftPair);
                    }
                }
                

                if(input[topPair.r][topPair.c] == 1 && !visited[topPair.r][topPair.c])
                queueForBFS.add(topPair);
            }
            if(currMin > indexesForPath.size())
            {
                indexesForShortestPath = indexesForPath;
            }
        }
        
        System.out.println("Shortest Path!");
        while (!indexesForShortestPath.isEmpty()) {
            PairInt index = indexesForShortestPath.pop();
            System.out.println(index.r + " " + index.c);
            
        }

        return input;

    }
    
    public LinkedListNode reverseLinkedList(LinkedListNode head) {
        if(head == null || head.next == null)
        return head;
        LinkedListNode prev = null, curr = head;
        while (curr != null) {
            LinkedListNode temp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = temp;
        }
        return prev;
    }

    public LinkedListNode getHead(){
        LinkedListNode head = new LinkedListNode();
        head.key = 1;
        head.next = new LinkedListNode();

        head.next.key = 2;
        head.next.next = new LinkedListNode();

        head.next.next.key = 3;
        head.next.next.next = new LinkedListNode();

        head.next.next.next.key = 4;
        head.next.next.next.next = new LinkedListNode();

        head.next.next.next.next.key = 5;
        return head;
    }

    public void printLinkedList(LinkedListNode head) {
        LinkedListNode temp = head;
        while (temp != null) {
            System.out.print(temp.key+" ");
            temp = temp.next;
        }
        System.out.println();
    }

    // input = "(ec(nt(ne))es)"
    public String removeParanthesis(String input) {
        Stack<Integer> leftParanthesisIndices = new Stack<>();
        char[] inputArr = input.toCharArray();
        char[] resultArrCopy = input.toCharArray();
        System.out.println(input);
        for (int i = 0; i < inputArr.length; i++) {
            if(input.charAt(i) == '(')
            leftParanthesisIndices.push(i);

            else if(input.charAt(i) == ')')
            {
                reverseArrayInPlace(resultArrCopy, leftParanthesisIndices.pop() + 1, i - 1);
                System.out.println(String.valueOf(resultArrCopy));
            }
        }

        return String.valueOf(resultArrCopy).replace("(", "").replace(")", "");
    }

    public void reverseArrayInPlace(char buffer[], int left, int right) {

        while (left < right)
        {
            char temp = buffer[left];
            buffer[left++] = buffer[right];
            buffer[right--] = temp;
        }
    }

    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        // nums2 = [228,231,34,225,28,222,128,53,50,247]
        // Having a simple suffix array would fail while maintaining a current right max look at 225, technically it should have 247 but with this approach nope!
        // Therefore use STACK!

        int result[] = new int[nums1.length];
        int largeArraySize = nums2.length;
        Map<Integer, Integer> numberNGE = new HashMap<>();
        
        Stack<Integer> stack = new Stack<>();
        stack.push(nums2[largeArraySize - 1]);
        numberNGE.put(nums2[largeArraySize - 1], -1);

        for (int i = largeArraySize - 2; i >= 0; i--) {
            if(nums2[i] < stack.peek()){
                numberNGE.put(nums2[i], stack.peek());
                stack.push(nums2[i]);
                continue;
            }

            while (!stack.isEmpty() && nums2[i] > stack.peek() ) {
                stack.pop();
            }

            if(stack.isEmpty()) {
                numberNGE.put(nums2[i], -1);
                stack.push(nums2[i]);
            }

            else {
                numberNGE.put(nums2[i], stack.peek());
                stack.push(nums2[i]);
            }
        }

        for (int i = 0; i < nums1.length; i++) {
            result[i] = numberNGE.get(nums1[i]);
        }

        return result;
    }

    public int[] nextGreaterElementsII(int[] nums) {
        int n = nums.length;
        int result[] = new int[n];
        Stack<Integer> stack = new Stack<>();
        result[n-1] = -1;
        stack.push(nums[n-1]);
        for(int i = 2*n - 2; i >= 0; i--) {
            int index = i%n;
            if(nums[index] < stack.peek()) {
                result[index] = stack.peek();
                stack.push(nums[index]);
                continue;
            }

            while(!stack.isEmpty() && stack.peek() < nums[index]) {
                stack.pop();
            }

            if(stack.isEmpty()) {
                result[index] = -1;
                stack.push(nums[index]);
            }

            else{
                result[index] = stack.peek();
                stack.push(nums[index]);
            }
        }

        return result;
    }

    public String gcdOfStrings(String str1, String str2) {
        if ((str1+str2).equals(str2+str1))
        return str1.substring(0, gcdEuclideanAlgo(str1.length(), str2.length()));

        return "";
    }

    public int gcdEuclideanAlgo(int a, int b){
        if(a == 0)
        return b;

        return gcdEuclideanAlgo(b % a, a);
    }

    public static void minimumBribes(List<Integer> q) {
        int i = 1;
        int result = 0;
        boolean isChaotic = false;
        while (i < q.size()) {
            int diff = q.get(i - 1) - q.get(i);
            if(diff == -1){
                i++;
                continue;
            }

            if(diff > -2)
            {
                isChaotic = true;
                break;
            }

            else{
                result += (diff * -1);
            }

            i = i + (diff * -1) + 1;
        }
    }

    static int minimumSwaps(int[] arr) {
        int arrCpy[] = Arrays.copyOf(arr, arr.length);
        Arrays.sort(arrCpy);
        Map<Integer, Integer> originalArrayIndexMap = new HashMap<>();
        for (int i = 0; i < arrCpy.length; i++) {
            originalArrayIndexMap.put(arr[i], i);
        }
        int result = 0;
        for (int i = 0; i < arrCpy.length; i++) {
            if(arrCpy[i] != arr[i]) {   
                result++;
                int prevValue = arr[i];
                swap(arr, i, originalArrayIndexMap.get(arrCpy[i]));   
                
                originalArrayIndexMap.put(prevValue, originalArrayIndexMap.get(arrCpy[i]));
                originalArrayIndexMap.put(arrCpy[i], i);
            }
        }
        return result;
    }
    
    static void swap(int[] arr, int src, int des){
        int temp = arr[src];
        arr[src] = arr[des];
        arr[des] = temp;
    }
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int arr[] = {5,3,2,1,39,44,-23,-19, -200};
        Test test = new Test();

        String[][] gradesTest = {{"J","80"},{"J","90"},{"K","80"},{"J","30"},{"K","60"},{"A","50"}};
        String[][] s1 = {{"Rohan", "84"}, {"Sachin", "102"}, {"Ishan", "55"}, {"Sachin", "18"}};
        // System.out.println(test.averageGrade(s1));

        // System.out.println(test.findPeakElement(arr));
        
        int[][] grid = {{0, 0, 0, 0, 5}, {0, 1, 1, 1, 0}, {2, 0, 0, 0, 0}};
        int[][] dp = new int[grid.length][grid[0].length];
        
        // Need to check with the interviewer if input could have negative numbers, if yes then fill the array with INT_MIN
        // for (int i = 0; i < input.length; i++) {
        //     Arrays.fill(dp[i], -1);
        // }
        // System.out.println(test.optimalPathRecDp(input, input.length - 1, 0, dp));

        // System.out.println(test.optimalPath(grid));
        // System.out.println(test.searchInRotatedArray(new int[]{3,4,5,1,2}, 10));

        // System.out.println(test.medianTwoSortedArrays(new int[]{2,3,5,15}, new int[]{1,3,4,7,10,12}));
        // System.out.println(test.medianTwoSortedArrays(new int[]{1,2}, new int[]{3,4}));

        // System.out.println(test.decimalConverstion(4, 333));
        
        Map<String, List<String>> graph = new HashMap<>();
        graph.put("King's Cross St Pancras", Arrays.asList("Angel","Russell Square"));
        graph.put("Angel", Arrays.asList("Old Street","Farringdon"));
        graph.put("Old Street", Arrays.asList("Barbican"));
        graph.put("Farringdon", Arrays.asList("Barbican", "St Paul's"));
        graph.put("Barbican", Arrays.asList("St Paul's"));
        graph.put("Moorgate", Arrays.asList("Holborn"));
        graph.put("Holborn", Arrays.asList("Chancery Lane"));
        graph.put("Chancery Lane", Arrays.asList("St Paul's"));
        graph.put("St Paul's", Arrays.asList("Bank"));
        
        // System.out.println(test.trainMap(graph, "King's Cross St Pancras", "Bank"));
        // System.out.println(shortestPath(graph, "King's Cross St Pancras", "Bank"));

        
        //child to parent relationship
        // Map<Integer, Integer> forest = new HashMap<>(){{
        //     put(1, 2);
        //     put(3, 4);
        //     put(7, 2);
        //     put(9, 1);
        // }};
        // System.out.println(test.longestTree(forest));

        // System.out.println(test.getSecondMinimumElement(arr));
        // System.out.println(test.powerOfANumber(2, 10));
        // System.out.println(test.firstNonRepeatingChar("aaa"));
        // arr = new int[]{1,2,3,4,5,6,4};
        // System.out.println(test.pivotIndexForArraySum(arr));
        // System.out.println(minimalSteps("ABABABAB"));
        // if (minimalSteps("ABABCABABCE") == 6
        //         && minimalSteps("ABCDABCE") == 8
        //         && minimalSteps("ABCABCE") == 5
        //         && minimalSteps("AAA") == 3
        //         && minimalSteps("AAAA") == 3
        //         && minimalSteps("BBB") == 3
        //         && minimalSteps("AAAAAA") == 4) {
        //     System.out.println();
        //     System.out.println("Pass");
        // } else {
        //     System.out.println();
        //     System.out.println("Fail");
        // }

        // System.out.println("abcd".substring(1,2));
        // int[] result = test.longestUniformSubString("abbcc");
        // System.out.println(result[0]+" "+result[1]);
        // test.primeFactors(121).forEach(a -> System.out.println(a));

        // System.out.println(test.pascalTriangleNumber(0, 0));
        // System.out.println(squareRoot(24,0.01)+" "+Math.sqrt(24));

        // System.out.println(test.election(1, 1));

        // System.out.println(test.atoi("-2") * test.atoi("-2"));
        
        // MedianFinder medianFinder =  new MedianFinder();
        // medianFinder.addNum(1);
        // medianFinder.addNum(2);
        // System.out.println(medianFinder.findMedian());
        // medianFinder.addNum(3);
        // System.out.println(medianFinder.findMedian());
        
        // LinkedListNode head = test.getHead();
        // LinkedListNode reversedHead = test.reverseLinkedList(head);
        // test.printLinkedList(reversedHead);

        int test_arr[] = {-1, -1, 2};

        // int[][] test_grid = {{0, 1, 1, 0, 0}, {1, 0, 1, 1, 0}, {1, 0, 0, 1, 0}};
        // int[][] test_grid = {{0, 1, 1, 0, 1}, {1, 0, 1, 1, 1}, {1, 0, 0, 1, 1}};

        // test.shortestPathBinaryMatrix(test_grid);
        // System.out.println();
        // int[] res = test.nextGreaterElementsII(test_arr);
        // String res = test.removeParanthesis("(ab)(cd)");
        // String res = test.removeParanthesis("(u(respect)i)");
        // System.out.println(res);
        
        System.out.println(test.subarraySum(test_arr, 2));

        sc.close();
    }
}

class LinkedListNode{
    int key;
    LinkedListNode next;
}

class Node {
    int key;
    Node left, right;

    public Node(int item)
    {
        key = item;
        left = right = null;
    }
}

class MedianFinder {

    private PriorityQueue<Integer> smallQueue;
    private PriorityQueue<Integer> largeQueue;
    public MedianFinder() {
        smallQueue = new PriorityQueue<>(Collections.reverseOrder());
        largeQueue = new PriorityQueue<>();
    }
    
    public void addNum(int num) {
        smallQueue.add(num);
        if(largeQueue.size() > 0 && smallQueue.peek() > largeQueue.peek()) {
            largeQueue.add(smallQueue.poll());
        }

        if(smallQueue.size() > largeQueue.size() + 1) {
            largeQueue.add(smallQueue.poll());
        }

        if(largeQueue.size() > smallQueue.size() + 1) {
            smallQueue.add(largeQueue.poll());
        }
    }
    
    public double findMedian() {
        if(smallQueue.size() > largeQueue.size()) {
            return smallQueue.peek();
        }

        if(largeQueue.size() > smallQueue.size()) {
            return largeQueue.peek();
        }

        return (largeQueue.peek() + smallQueue.peek()) * 1.0 / 2;
    }
}