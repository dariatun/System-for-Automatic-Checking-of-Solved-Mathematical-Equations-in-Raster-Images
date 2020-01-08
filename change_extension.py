file_read = open('/Users/dariatunina/mach-lerinig/test-eq1.txt', 'r')
file_write = open('/Users/dariatunina/mach-lerinig/test-txt.txt', 'w+')

for line in file_read:
    file_write.write(line[0:-4] + 'txt')
    file_write.write('\n')
file_read.close()
file_write.close()
print('Finished')
