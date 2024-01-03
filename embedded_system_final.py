from pybleno import Bleno, BlenoPrimaryService, BlenoCharacteristic, BlenoDescriptor
import sys
import signal

bleno = Bleno()
name = 'MyESP32'
service_uuid = '4fafc201-1fb5-459e-8fcc-c5c9c331914b'
characteristic_uuid = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'

class MyCharacteristic(BlenoCharacteristic):
    def __init__(self):
        BlenoCharacteristic.__init__(self, {
            'uuid': characteristic_uuid,
            'properties': ['read', 'write', 'notify', 'indicate'],
            'value': None
        })

    def onReadRequest(self, offset, callback):
        print("Read request received")
        callback(BlenoCharacteristic.RESULT_SUCCESS, "Hello Client!")

    def onWriteRequest(self, data, offset, withoutResponse, callback):
        print("Write request: %s" % data)
        callback(BlenoCharacteristic.RESULT_SUCCESS)

class MyService(BlenoPrimaryService):
    def __init__(self):
        BlenoPrimaryService.__init__(self, {
            'uuid': service_uuid,
            'characteristics': [
                MyCharacteristic()
            ]
        })

def on_start_success(state):
    print('on -> start: state = %s' % (state))
    if (state == 'poweredOn'):
        bleno.startAdvertising(name, [service_uuid])
    else:
        bleno.stopAdvertising()

bleno.on('stateChange', on_start_success)

def on_advertising_start(error):
    print('on -> advertisingStart: %s' % ('error ' + error if error else 'success'))

    if not error:
        bleno.setServices([
            MyService()
        ])

bleno.on('advertisingStart', on_advertising_start)

bleno.start()

print('Hit <ENTER> to disconnect')
input()

bleno.stopAdvertising()
bleno.disconnect()

print('terminated.')
sys.exit(1)